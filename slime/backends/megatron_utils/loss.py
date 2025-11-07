from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any, Dict, Tuple, Union

import torch
from megatron.core import mpu

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_policy_loss,
    get_advantages_and_returns,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.types import RolloutBatch

from .cp_utils import (all_gather_with_cp,get_logits_and_tokens_offset_with_cp,get_sum_of_sample_mean,slice_log_prob_with_cp)


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    assert logits.size(0) == 1, f"{logits.shape}"
    assert logits.dtype == torch.float32, f"{logits.dtype}"

    def _dump_non_finite(
        prefix: str,
        tensor: torch.Tensor,
        sample_idx: int,
        total_length: int,
        response_length: int,
        extra: dict | None = None,
        tokens: torch.Tensor | None = None,
    ):
        try:
            mask = ~torch.isfinite(tensor)
            num_bad = int(mask.sum().item())
            first_pos = mask.nonzero(as_tuple=False)[0].tolist() if num_bad > 0 else None

            tensor_cpu = tensor.detach().float().cpu()
            finite_tensor_cpu = torch.where(
                torch.isfinite(tensor_cpu), tensor_cpu, torch.zeros_like(tensor_cpu)
            )
            max_abs = finite_tensor_cpu.abs().max().item()

            token_stats = None
            if tokens is not None and tokens.numel() > 0:
                tokens_cpu = tokens.detach().long().cpu()
                token_stats = {
                    "min": int(tokens_cpu.min().item()),
                    "max": int(tokens_cpu.max().item()),
                    "mean": float(tokens_cpu.float().mean().item()),
                }

            rank = -1
            dist_module = getattr(torch, "distributed", None)
            if dist_module is not None and dist_module.is_available() and dist_module.is_initialized():
                rank = dist_module.get_rank()

            payload = {
                "prefix": prefix,
                "sample_idx": sample_idx,
                "total_length": total_length,
                "response_length": response_length,
                "num_bad": num_bad,
                "first_bad_position": first_pos,
                "max_abs": max_abs,
                "extra": extra,
                "token_stats": token_stats,
            }

            path = f"/tmp/stability_bad_logits_rank{rank}_sample{sample_idx}_{prefix.replace(' ', '_')}.pt"
            payload["tensor"] = tensor_cpu
            if tokens is not None:
                payload["tokens"] = tokens.detach().cpu()
            torch.save(payload, path)
            print(
                f"[Training Stability Debug] Saved non-finite tensor snapshot to {path} "
                f"(prefix={prefix}, num_bad={num_bad}, first_bad_position={first_pos}, max_abs={max_abs})",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - best-effort debug aid
            print(
                f"[Training Stability Debug] Failed to dump non-finite tensor (prefix={prefix}, sample_idx={sample_idx}): {exc}",
                flush=True,
            )

    logits = logits.squeeze(0)
    logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    log_probs_list = []
    if with_entropy:
        entropy_list = []
    end = 0
    for sample_idx, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths)
    ):
        if cp_size == 1:
            end += total_length
            start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]

            sanitized = False
            if not torch.isfinite(logits_chunk).all():
                _dump_non_finite(
                    "chunk_0_non_cp",
                    logits_chunk,
                    sample_idx,
                    total_length,
                    response_length,
                    extra={
                        "start": start,
                        "end": end,
                        "path": "non_cp",
                    },
                    tokens=tokens_chunk,
                )
                logits_chunk = torch.nan_to_num(
                    logits_chunk, nan=0.0, posinf=1e4, neginf=-1e4
                )
                sanitized = True

            tp_world_size = max(mpu.get_tensor_model_parallel_world_size(), 1)
            local_vocab_size = logits_chunk.size(-1)
            global_vocab_upper_bound = local_vocab_size * tp_world_size
            token_max_val = tokens_chunk.max().item()
            token_min_val = tokens_chunk.min().item()
            if token_max_val >= global_vocab_upper_bound or token_min_val < 0:
                print(
                    "[Training Stability] Token index out of bounds detected "
                    f"(sample_idx={sample_idx}, global_vocab_upper_bound={global_vocab_upper_bound}, "
                    f"token_min={token_min_val}, token_max={token_max_val}, "
                    f"total_length={total_length}, response_length={response_length})"
                )
                raise RuntimeError("Token index out of vocabulary range before log prob computation.")

            if sanitized:
                log_prob = logits.new_zeros(response_length)
                entropy = logits.new_zeros(response_length) if with_entropy else None
            else:
                log_prob, entropy = calculate_log_probs_and_entropy(
                    logits_chunk, tokens_chunk, mpu.get_tensor_model_parallel_group(), with_entropy=with_entropy
                )
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            sanitized = False
            if not torch.isfinite(logits_0).all():
                _dump_non_finite(
                    "chunk_0_cp",
                    logits_0,
                    sample_idx,
                    total_length,
                    response_length,
                    extra={"part": 0, "path": "cp"},
                    tokens=tokens_0,
                )
                logits_0 = torch.nan_to_num(logits_0, nan=0.0, posinf=1e4, neginf=-1e4)
                sanitized = True
            if not torch.isfinite(logits_1).all():
                _dump_non_finite(
                    "chunk_1_cp",
                    logits_1,
                    sample_idx,
                    total_length,
                    response_length,
                    extra={"part": 1, "path": "cp"},
                    tokens=tokens_1,
                )
                logits_1 = torch.nan_to_num(logits_1, nan=0.0, posinf=1e4, neginf=-1e4)
                sanitized = True

            tp_world_size = max(mpu.get_tensor_model_parallel_world_size(), 1)
            local_vocab_size = logits_0.size(-1)
            global_vocab_upper_bound = local_vocab_size * tp_world_size
            token_max_candidates = []
            token_min_candidates = []
            if tokens_0.numel() > 0:
                token_max_candidates.append(tokens_0.max())
                token_min_candidates.append(tokens_0.min())
            if tokens_1.numel() > 0:
                token_max_candidates.append(tokens_1.max())
                token_min_candidates.append(tokens_1.min())

            if token_max_candidates:
                token_max = torch.stack(token_max_candidates).max().item()
                token_min = torch.stack(token_min_candidates).min().item()
                if token_max >= global_vocab_upper_bound or token_min < 0:
                    print(
                        "[Training Stability] Token index out of bounds detected (CP path) "
                        f"(sample_idx={sample_idx}, global_vocab_upper_bound={global_vocab_upper_bound}, "
                        f"token_min={token_min}, token_max={token_max}, "
                        f"total_length={total_length}, response_length={response_length})"
                    )
                    raise RuntimeError("Token index out of vocabulary range before log prob computation.")

            if sanitized:
                log_prob = logits.new_zeros(response_length)
                entropy = logits.new_zeros(response_length) if with_entropy else None
            else:
                log_prob_0, entropy_0 = calculate_log_probs_and_entropy(
                    logits_0,
                    tokens_0,
                    mpu.get_tensor_model_parallel_group(),
                    with_entropy=with_entropy,
                )
                log_prob_1, entropy_1 = calculate_log_probs_and_entropy(
                    logits_1,
                    tokens_1,
                    mpu.get_tensor_model_parallel_group(),
                    with_entropy=with_entropy,
                )
                log_prob = torch.cat([log_prob_0, log_prob_1], dim=0)
                if with_entropy:
                    entropy = torch.cat([entropy_0, entropy_1], dim=0)

        yield logits_chunk, tokens_chunk


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    log_probs_list = []
    entropy_list = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk, tokens_chunk, mpu.get_tensor_model_parallel_group(), with_entropy=with_entropy
        )

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    return res


def compute_advantages_and_returns(args, rollout_data):
    log_probs: list[torch.Tensor] | None = rollout_data.get("log_probs", None)
    rollout_log_probs: list[torch.Tensor] | None = rollout_data.get("rollout_log_probs", None)
    ref_log_probs: list[torch.Tensor] | None = rollout_data.get("ref_log_probs", None)
    rewards: list[float] | None = rollout_data.get("rewards", None)
    values: Union[None, list[torch.Tensor]] = rollout_data.get("values", None)
    response_lengths: list[int] | None = rollout_data.get("response_lengths", None)
    loss_masks: list[torch.Tensor] | None = rollout_data.get("loss_masks", None)
    total_lengths: list[int] | None = rollout_data.get("total_lengths", None)

    if log_probs is None or loss_masks is None or response_lengths is None or total_lengths is None:
        return

    use_is_distill_loss = getattr(args, "enable_on_policy_distill", False) and getattr(
        args, "use_is_distill_loss", False
    )

    if use_is_distill_loss:
        _compute_is_distill_advantages(
            args=args,
            rollout_data=rollout_data,
            rollout_log_probs=rollout_log_probs,
            teacher_log_probs=ref_log_probs,
            rewards=rewards,
            loss_masks=loss_masks,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
        )
        return

    reverse_kl_stats: list[float] | None = None
    if args.advantage_estimator == "reverse_kl":
        if ref_log_probs is None:
            raise ValueError("On-policy distillation requires teacher log probabilities (ref_log_probs).")
        if rewards is None:
            raise ValueError("reverse_kl advantage estimator requires rollout rewards for baseline computation.")

        if not getattr(args, "balance_data", False):
            raise ValueError(
                "reverse_kl advantage estimator requires --balance-data to keep prompt groups intact."
            )

        device = loss_masks[0].device
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        group_size = max(int(getattr(args, "n_samples_per_prompt", 1)), 1)

        if rewards_tensor.numel() % group_size != 0:
            raise ValueError(
                f"Number of rewards {rewards_tensor.numel()} is not divisible by "
                f"group size n_samples_per_prompt={group_size}."
            )

        rewards_reshaped = rewards_tensor.view(-1, group_size)
        centered_rewards = rewards_reshaped - rewards_reshaped.mean(dim=1, keepdim=True)
        reward_scalars = centered_rewards.reshape(-1).tolist()

        advantages: list[torch.Tensor] = []
        returns: list[torch.Tensor] = []
        reverse_kl_stats = []
        reward_advantages: list[torch.Tensor] = []
        cp_size = mpu.get_context_parallel_world_size()
        kl_coef = getattr(args, "kl_penalty_coef", getattr(args, "kl_coef", 0.0))
        debug_printed = True

        for idx, (student_lp_raw, teacher_lp_raw, mask_tensor, total_len, resp_len) in enumerate(
            zip(rollout_log_probs, ref_log_probs, loss_masks, total_lengths, response_lengths)
        ):
            mask_local = mask_tensor
            if cp_size > 1:
                mask_local = slice_log_prob_with_cp(mask_tensor, total_len, resp_len)
                if isinstance(mask_local, list):
                    mask_local = torch.tensor(mask_local, dtype=mask_tensor.dtype, device=mask_tensor.device)

            if mask_local.numel() == 0 or student_lp_raw.numel() == 0:
                print(
                    f"[Distill] Skip sample {idx} due to empty tensors "
                    f"(len mask={mask_local.numel()}, student={student_lp_raw.numel()})."
                )
                continue

            student_lp = student_lp_raw
            teacher_lp = teacher_lp_raw
            if isinstance(student_lp, list):
                student_lp = torch.tensor(student_lp, dtype=torch.float32, device=device)
            if isinstance(teacher_lp, list):
                teacher_lp = torch.tensor(teacher_lp, dtype=torch.float32, device=device)

            if student_lp.numel() != teacher_lp.numel() or student_lp.numel() != mask_local.numel():
                raise RuntimeError(
                    f"[Distill] CP slicing mismatch for sample {idx}: "
                    f"student={student_lp.numel()}, teacher={teacher_lp.numel()}, mask={mask_local.numel()}"
                )

            mask_float = mask_local.to(dtype=torch.float32, device=device)
            reward_component = mask_float * reward_scalars[idx]
            reward_advantages.append(reward_component)

            student_lp = student_lp.to(device=device, dtype=torch.float32)
            teacher_lp = teacher_lp.to(device=device, dtype=torch.float32)
            reverse_kl = (student_lp - teacher_lp) * mask_float
            ##############################################################################################################################
            composite_adv = reward_component - kl_coef * reverse_kl
            advantages.append(composite_adv)
            returns.append(reward_component.clone())

            denom = torch.clamp_min(mask_float.sum(), 1.0)
            sample_mean = reverse_kl.sum() / denom
            reverse_kl_stats.append(float(sample_mean.detach().cpu()))

            if (
                not debug_printed
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.is_pipeline_last_stage()
            ):
                student_dbg = student_lp[:100].detach().float().cpu().tolist()
                teacher_dbg = teacher_lp[:100].detach().float().cpu().tolist()
                print(f"[Distill] Sample {idx} student_lp[:100]={student_dbg}")
                print(f"[Distill] Sample {idx} teacher_lp[:100]={teacher_dbg}")
                debug_printed = True

        kl = None
    else:
        if args.kl_coef == 0:
            kl = [
                torch.zeros_like(
                    log_probs[i],
                    dtype=torch.float32,
                    device=log_probs[i].device,
                )
                for i in range(len(log_probs))
            ]
        else:
            kl = [
                compute_approx_kl(
                    log_probs[i],
                    ref_log_probs[i],
                    kl_loss_type=args.kl_loss_type,
                )
                for i in range(len(log_probs))
            ]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)

        if args.advantage_estimator in ["grpo", "gspo"]:
            returns = get_grpo_returns(rewards, kl)
            advantages = [r for r in returns]

        elif args.advantage_estimator == "reinforce_plus_plus":
            returns = get_reinforce_plus_plus_returns(
                rewards=rewards,
                kl=kl,
                loss_masks=loss_masks,
                response_lengths=response_lengths,
                total_lengths=total_lengths,
                kl_coef=args.kl_coef,
                gamma=args.gamma,
            )
            advantages = [r for r in returns]

        elif args.advantage_estimator == "reinforce_plus_plus_baseline":
            advantages = get_reinforce_plus_plus_baseline_advantages(
                rewards=rewards,
                kl=kl,
                loss_masks=loss_masks,
                kl_coef=args.kl_coef,
            )
            returns = advantages

        else:
            raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(total_len, response_len)

                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
    if reverse_kl_stats is not None:
        rollout_data["reverse_kl"] = reverse_kl_stats


def _compute_is_distill_advantages(
    *,
    args,
    rollout_data: dict,
    rollout_log_probs: list[torch.Tensor] | None,
    teacher_log_probs: list[torch.Tensor] | None,
    rewards: list[float] | None,
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
) -> None:
    """
    Construct composite on-policy distillation advantages, mirroring Tinker Cookbook:

        A_t = A_t^{reward} - kl_coef * (log p_student,t - log p_teacher,t)
    """
    if rollout_log_probs is None or teacher_log_probs is None:
        raise ValueError(
            "On-policy distillation with importance-sampling loss requires teacher and rollout log probabilities."
        )

    device = loss_masks[0].device
    cp_size = mpu.get_context_parallel_world_size()

    # ----- reward component -----
    reward_advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    if rewards is None or args.advantage_estimator == "reverse_kl":
        for mask in loss_masks:
            zeros = torch.zeros_like(mask, dtype=torch.float32, device=device)
            reward_advantages.append(zeros)
            returns.append(zeros.clone())
    else:
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        if args.advantage_estimator in ["grpo", "gspo"]:
            returns = get_grpo_returns(rewards_tensor, None)
            reward_advantages = [ret.clone().to(device=device) for ret in returns]
        elif args.advantage_estimator == "reinforce_plus_plus":
            returns = get_reinforce_plus_plus_returns(
                rewards=rewards_tensor,
                kl=None,
                loss_masks=loss_masks,
                response_lengths=response_lengths,
                total_lengths=total_lengths,
                kl_coef=0.0,
                gamma=args.gamma,
            )
            reward_advantages = [ret.clone().to(device=device) for ret in returns]
        elif args.advantage_estimator == "reinforce_plus_plus_baseline":
            reward_advantages = get_reinforce_plus_plus_baseline_advantages(
                rewards=rewards_tensor,
                kl=None,
                loss_masks=loss_masks,
                kl_coef=0.0,
            )
            reward_advantages = [adv.to(device=device) for adv in reward_advantages]
            returns = [adv.clone() for adv in reward_advantages]
        else:
            raise NotImplementedError(
                f"advantage_estimator '{args.advantage_estimator}' is not supported with IS distillation."
            )

    # ----- KL component -----
    kl_coef = getattr(args, "kl_penalty_coef", getattr(args, "kl_coef", 0.0))
    composite_advantages: list[torch.Tensor] = []
    reverse_kl_stats: list[float] = []

    def _slice_tensor_with_cp(tensor: torch.Tensor, total_len: int, resp_len: int) -> torch.Tensor:
        if cp_size == 1:
            return tensor

        prompt_len = total_len - resp_len
        _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(total_len, resp_len)

        slices = []
        for start, end in tokens_offset:
            if end <= start:
                continue
            offset_start = max(start - prompt_len, 0)
            offset_end = max(end - prompt_len, 0)
            if offset_end > offset_start:
                slices.append(tensor[offset_start:offset_end])
        if not slices:
            return torch.zeros(0, dtype=tensor.dtype, device=tensor.device)
        return torch.cat(slices, dim=0)

    for idx, reward_adv in enumerate(reward_advantages):
        reward_adv = reward_adv.to(device=device, dtype=torch.float32)
        mask = loss_masks[idx].to(device=device)
        mask_float = mask.to(dtype=torch.float32)

        if reward_adv.numel() != mask_float.numel():
            raise RuntimeError(
                f"[Distill] Reward advantage/mask mismatch for sample {idx}: "
                f"adv={reward_adv.size()}, mask={mask_float.size()}"
            )

        if kl_coef != 0.0:
            total_len = total_lengths[idx]
            resp_len = response_lengths[idx]

            student_lp_raw = rollout_log_probs[idx]
            teacher_lp_raw = teacher_log_probs[idx]

            if cp_size > 1:
                student_lp_full = all_gather_with_cp(student_lp_raw, total_len, resp_len)
                teacher_lp_full = all_gather_with_cp(teacher_lp_raw, total_len, resp_len)
                student_lp = student_lp_full[-resp_len:]
                teacher_lp = teacher_lp_full[-resp_len:]
            else:
                student_lp = student_lp_raw[-resp_len:] if student_lp_raw.numel() != resp_len else student_lp_raw
                teacher_lp = teacher_lp_raw[-resp_len:] if teacher_lp_raw.numel() != resp_len else teacher_lp_raw

            if isinstance(student_lp, list):
                student_lp = torch.tensor(student_lp, dtype=torch.float32, device=device)
            if isinstance(teacher_lp, list):
                teacher_lp = torch.tensor(teacher_lp, dtype=torch.float32, device=device)

            if student_lp.numel() != teacher_lp.numel() or student_lp.numel() != reward_adv.numel():
                raise RuntimeError(
                    f"[Distill] KL shapes mismatch for sample {idx}: "
                    f"student={student_lp.shape}, teacher={teacher_lp.shape}, reward_adv={reward_adv.shape}"
                )

            student_lp = student_lp.to(device=device, dtype=torch.float32)
            teacher_lp = teacher_lp.to(device=device, dtype=torch.float32)

            kl_delta = (student_lp - teacher_lp) * mask_float
            denom = torch.clamp_min(mask_float.sum(), 1.0)
            reverse_kl_stats.append(float((kl_delta.sum() / denom).detach().cpu()))

            composite_adv = reward_adv - kl_coef * kl_delta
        else:
            reverse_kl_stats.append(0.0)
            composite_adv = reward_adv

        if cp_size > 1:
            total_len = total_lengths[idx]
            resp_len = response_lengths[idx]
            reward_adv = _slice_tensor_with_cp(reward_adv, total_len, resp_len)
            composite_adv = _slice_tensor_with_cp(composite_adv, total_len, resp_len)

        composite_advantages.append(composite_adv)
        reward_advantages[idx] = reward_adv

    # ----- optional whitening -----
    if args.normalize_advantages and composite_advantages:
        all_advs = torch.cat(composite_advantages)
        if cp_size == 1:
            all_masks = torch.cat(loss_masks).to(device=device)
        else:
            mask_chunks = []
            for i in range(len(composite_advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(total_len, response_len)

                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                full_mask = loss_masks[i].to(device=device)
                local_parts = []
                if res_e0 > res_s0:
                    local_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_parts.append(full_mask[res_s1:res_e1])
                mask_chunks.append(torch.cat(local_parts) if local_parts else torch.tensor([], device=device))
            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            whitened = distributed_masked_whiten(all_advs, all_masks.to(all_advs.device), shift_mean=True)
            chunk_lengths = [adv.size(0) for adv in composite_advantages]
            composite_advantages = list(torch.split(whitened, chunk_lengths))

    if cp_size > 1 and returns:
        returns = [
            _slice_tensor_with_cp(ret.to(device=device, dtype=torch.float32), total_lengths[idx], response_lengths[idx])
            for idx, ret in enumerate(returns)
        ]

    rollout_data["reward_advantages"] = reward_advantages
    rollout_data["advantages"] = composite_advantages
    rollout_data["returns"] = returns
    rollout_data["reverse_kl"] = reverse_kl_stats


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Temporal Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    entropy = log_probs_and_entropy["entropy"]

    # Apply high-entropy token filtering if enabled
    if getattr(args, 'high_entropy_token_filter', False):
        entropy_percentile = getattr(args, 'entropy_percentile', 0.2)

        # Concatenate all entropies and masks
        all_entropy = torch.cat(entropy, dim=0)
        loss_masks = batch["loss_masks"]

        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(entropy)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(total_len, response_len)

                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                local_mask_chunk = (
                    torch.cat(local_mask_parts) if local_mask_parts
                    else torch.tensor([], device=all_entropy.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)
            all_masks = torch.cat(mask_chunks)

        # Compute entropy threshold from valid tokens only
        if all_masks.sum() > 0:
            valid_entropy = all_entropy[all_masks.bool()]
            entropy_threshold = torch.quantile(valid_entropy, 1.0 - entropy_percentile)

            # Create high-entropy mask
            high_entropy_mask = (all_entropy >= entropy_threshold).float()

            # Update loss_masks
            chunk_lengths = [ent.size(0) for ent in entropy]
            high_entropy_chunks = list(torch.split(high_entropy_mask, chunk_lengths))
            batch["loss_masks"] = [
                loss_mask * high_entropy_chunk
                for loss_mask, high_entropy_chunk in zip(loss_masks, high_entropy_chunks)
            ]

            # Recompute sum_of_sample_mean with updated masks
            sum_of_sample_mean = get_sum_of_sample_mean(
                total_lengths, response_lengths, batch["loss_masks"], args.calculate_per_token_loss
            )

    if getattr(args, "enable_on_policy_distill", False) and getattr(args, "use_is_distill_loss", False):
        return _importance_sampling_policy_loss(
            args=args,
            batch=batch,
            current_log_probs=log_probs,
            sum_of_sample_mean=sum_of_sample_mean,
        )

    old_log_probs_flat: torch.Tensor | None = None
    if args.advantage_estimator == "gspo":
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(log_probs, total_lengths, response_lengths)
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(old_log_probs, total_lengths, response_lengths)
        ]

        loss_masks = batch["loss_masks"]
        ppo_kl = [
            ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
            for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks)
        ]
        ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, log_probs)]
        ppo_kl = torch.cat(ppo_kl, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        old_log_probs_flat = torch.cat(full_old_log_probs, dim=0)
    else:
        old_log_probs_flat = torch.cat(batch["log_probs"], dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs_flat - log_probs

    cispo_stats: dict[str, torch.Tensor] | None = None
    is_cispo = getattr(args, "policy_objective", "ppo") == "cispo"

    if is_cispo:
        assert old_log_probs_flat is not None, "old log probs must be available for CISPO."
        ratio = torch.exp(log_probs - old_log_probs_flat)
        ratio_clipped = ratio
        eps_low = getattr(args, "cispo_eps_low", 0.0)
        eps_high = getattr(args, "cispo_eps_high", 2.0)
        if eps_low is not None and eps_low > 0.0:
            ratio_clipped = ratio_clipped.clamp_min(1.0 - eps_low)
        if eps_high is not None and eps_high > 0.0:
            ratio_clipped = ratio_clipped.clamp_max(1.0 + eps_high)
        clip_mask = (ratio_clipped != ratio).float()
        pg_loss = -(ratio_clipped.detach() * advantages) * log_probs
        pg_clipfrac = clip_mask
        cispo_stats = {
            "ratio_mean": sum_of_sample_mean(ratio_clipped.detach()),
            "ratio_max": ratio.detach().max(),
        }
    else:
        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)


    # Apply off-policy correction using importance sampling if enabled
    if args.use_tis:

        def vanilla_tis_function(
            args,
            *,
            pg_loss: torch.Tensor,
            train_log_probs: list[torch.Tensor],
            rollout_log_probs: list[torch.Tensor],
            loss_masks: list[torch.Tensor],
            **kwargs: Any,
        ) -> Tuple[torch.Tensor, list[torch.Tensor], Dict[str, torch.Tensor]]:
            rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
            old_log_probs = torch.cat(train_log_probs, dim=0)
            tis = torch.exp(old_log_probs - rollout_log_probs)
            tis_abs = torch.exp((old_log_probs - rollout_log_probs).abs())
            tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
            tis_clipfrac = (tis_weights != tis).float()
            metrics = {
                "tis": tis.clone().detach(),
                "tis_clipfrac": tis_clipfrac.clone().detach(),
                "tis_abs": tis_abs.clone().detach(),
            }
            pg_loss = pg_loss * tis_weights
            return pg_loss, loss_masks, metrics

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
    # Apply TIS off-policy correction using importance sampling if enabled
    tis_metrics = {}
    if args.use_tis:

        def vanilla_tis_function(
            args,
            *,
            train_log_probs,
            rollout_log_probs,
            **kwargs,
        ):
            rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
            old_log_probs = torch.cat(train_log_probs, dim=0)
            tis = torch.exp(old_log_probs - rollout_log_probs)
            tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
            tis_clipfrac = (tis_weights != tis).float()
            metrics = {
                "tis": tis.clone().detach(),
                "tis_clipfrac": tis_clipfrac.clone().detach(),
            }
            return tis_weights, metrics

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }
        tis_clip = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
        tis_clipfrac = (tis_clip != tis).float()

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        tis_weights, tis_metrics = tis_func(**tis_kwargs)

        pg_loss = pg_loss * tis_weights

    pg_loss = sum_of_sample_mean(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    if cispo_stats is not None:
        cispo_stats["clipfrac"] = pg_clipfrac.clone().detach()

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.use_tis:
        reported_loss["ois"] = sum_of_sample_mean(ois).clone().detach()
        for metric_key, metric_value in tis_metrics.items():
            reported_loss[metric_key] = sum_of_sample_mean(metric_value).clone().detach()

    if cispo_stats is not None:
        reported_loss["cispo_ratio_mean"] = cispo_stats["ratio_mean"].clone().detach()
        reported_loss["cispo_ratio_max"] = cispo_stats["ratio_max"].clone().detach()
        reported_loss["cispo_clipfrac"] = cispo_stats["clipfrac"].clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
    )

    log_probs_list = log_probs_and_entropy["log_probs"]

    non_finite_info = []
    for sample_idx, (log_prob_tensor, resp_len, tot_len, loss_mask) in enumerate(
        zip(
            log_probs_list,
            batch.get("response_lengths", []),
            batch.get("total_lengths", []),
            batch.get("loss_masks", []),
        )
    ):
        if not torch.isfinite(log_prob_tensor).all():
            non_finite_values = log_prob_tensor[~torch.isfinite(log_prob_tensor)]
            non_finite_info.append(
                {
                    "idx": sample_idx,
                    "count": non_finite_values.numel(),
                    "min": non_finite_values.min().item()
                    if non_finite_values.numel() > 0
                    else float("nan"),
                    "max": non_finite_values.max().item()
                    if non_finite_values.numel() > 0
                    else float("nan"),
                    "response_length": resp_len,
                    "total_length": tot_len,
                    "loss_mask_sum": loss_mask.sum().item()
                    if isinstance(loss_mask, torch.Tensor)
                    else None,
                }
            )

    log_probs = torch.cat(log_probs_list, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    if not torch.isfinite(loss) or non_finite_info:
        loss_mask_sums = (
            [mask.sum().item() for mask in batch.get("loss_masks", [])]
            if isinstance(batch.get("loss_masks", None), list)
            else None
        )
        print(
            "[Training Stability] Non-finite loss detected. "
            f"loss={loss}, "
            f"num_log_probs={log_probs.numel()}, "
            f"loss_mask_sums={loss_mask_sums}, "
            f"response_lengths={batch.get('response_lengths', None)}, "
            f"total_lengths={batch.get('total_lengths', None)}, "
            f"non_finite_info={non_finite_info}"
        )
        raise RuntimeError("Encountered non-finite loss during SFT training.")

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
    )

    loss_function_kwargs = {
        "args": args,
        "batch": batch,
        "logits": logits,
        "sum_of_sample_mean": sum_of_sample_mean,
    }

    match args.loss_type:
        case "policy_loss":
            loss, log = policy_loss_function(**loss_function_kwargs)
        case "value_loss":
            loss, log = value_loss_function(**loss_function_kwargs)
        case "sft_loss":
            loss, log = sft_loss_function(**loss_function_kwargs)
        case "custom_loss":
            custom_loss_function = load_function(args.custom_loss_function_path)
            loss, log = custom_loss_function(**loss_function_kwargs)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    loss = (
        loss * num_microbatches / args.global_batch_size * mpu.get_data_parallel_world_size(with_context_parallel=True)
    )

    return (
        loss,
        num_tokens if args.calculate_per_token_loss else 1,
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else num_tokens,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )


def _importance_sampling_policy_loss(*, args, batch, current_log_probs, sum_of_sample_mean):
    """
    Importance-sampling policy gradient used for on-policy distillation:

        L = - Σ_t mask_t · A_t · log p_θ(x_t | x_{<t})

    where advantages already encode reward and KL components.
    """
    advantages = batch["advantages"]
    loss_masks = batch["loss_masks"]
    total_lengths = batch["total_lengths"]
    response_lengths = batch["response_lengths"]

    cp_size = mpu.get_context_parallel_world_size()
    device = current_log_probs[0].device

    per_token_terms: list[torch.Tensor] = []
    adv_sum = torch.tensor(0.0, device=device)
    weight_sum = torch.tensor(0.0, device=device)

    for idx, (adv, logprob, mask) in enumerate(zip(advantages, current_log_probs, loss_masks)):
        adv_local = adv.to(device=device, dtype=logprob.dtype)
        mask_tensor = mask
        if cp_size > 1:
            mask_tensor = slice_log_prob_with_cp(mask, total_lengths[idx], response_lengths[idx])
            if isinstance(mask_tensor, list):
                mask_tensor = torch.tensor(mask_tensor, device=device, dtype=logprob.dtype)
        mask_float = mask_tensor.to(device=device, dtype=logprob.dtype)

        if adv_local.numel() != mask_float.numel() or logprob.numel() != mask_float.numel():
            raise RuntimeError(
                f"[Distill] Advantage/logprob mismatch for sample {idx}: "
                f"adv={adv_local.size()}, logprob={logprob.size()}, mask={mask_float.size()}"
            )

        weighted_adv = adv_local * mask_float
        per_token_terms.append(weighted_adv * logprob)
        adv_sum += weighted_adv.sum()
        weight_sum += mask_float.sum()

    if per_token_terms:
        loss_tensor = torch.cat(per_token_terms, dim=0)
        loss = -sum_of_sample_mean(loss_tensor)
    else:
        loss = torch.tensor(0.0, device=device)


    avg_adv = (adv_sum / torch.clamp_min(weight_sum, 1.0)).detach().cpu().item()
    token_count = weight_sum.detach().cpu().item()

    log = {
        "policy/adv_mean": avg_adv,
        "policy/token_count": token_count,
    }
    return loss, log
