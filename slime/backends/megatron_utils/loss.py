from typing import Union

import torch
from megatron.core import mpu

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_policy_loss,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)

from .cp_utils import all_gather_with_cp, get_logits_and_tokens_offset_with_cp, get_sum_of_sample_mean


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
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

            path = f"/tmp/q_tuning_bad_logits_rank{rank}_sample{sample_idx}_{prefix.replace(' ', '_')}.pt"
            payload["tensor"] = tensor_cpu
            if tokens is not None:
                payload["tokens"] = tokens.detach().cpu()
            torch.save(payload, path)
            print(
                f"[Q-Tuning Debug] Saved non-finite tensor snapshot to {path} "
                f"(prefix={prefix}, num_bad={num_bad}, first_bad_position={first_pos}, max_abs={max_abs})",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - best-effort debug aid
            print(
                f"[Q-Tuning Debug] Failed to dump non-finite tensor (prefix={prefix}, sample_idx={sample_idx}): {exc}",
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
                    "[Q-Tuning] Token index out of bounds detected "
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
                        "[Q-Tuning] Token index out of bounds detected (CP path) "
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

            end += 2 * chunk_size

        log_probs_list.append(log_prob.squeeze(-1))
        if with_entropy:
            entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    return res


def compute_advantages_and_returns(args, rollout_data):
    log_probs: list[torch.Tensor] = rollout_data.get("log_probs", None)
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs", None)
    rewards: list[float] = rollout_data.get("rewards", None)
    values: Union[None, list[torch.Tensor]] = rollout_data.get("values", None)
    response_lengths: list[int] = rollout_data.get("response_lengths", None)
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks", None)
    total_lengths: list[int] = rollout_data.get("total_lengths", None)

    if log_probs is None:
        return

    if args.kl_coef == 0:
        # when kl_coef is 0, we won't compute ref_log_prob
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
        # TODO: is the copy necessary?
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

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
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

                # Convert global offsets to response-space offsets
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

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
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

            whitened_advs_flat = distributed_masked_whiten(all_advs, all_masks, shift_mean=True)
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def policy_loss_function(args, batch, logits, sum_of_sample_mean):
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["log_probs"]

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
    else:
        old_log_probs = torch.cat(batch["log_probs"], dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

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

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        tis_weights, tis_metrics = tis_func(**tis_kwargs)

        pg_loss = pg_loss * tis_weights

    pg_loss = sum_of_sample_mean(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

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

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.use_tis:
        reported_loss["ois"] = sum_of_sample_mean(ois).clone().detach()
        for metric_key, metric_value in tis_metrics.items():
            reported_loss[metric_key] = sum_of_sample_mean(metric_value).clone().detach()

    return loss, reported_loss


def sft_loss_function(args, batch, logits, sum_of_sample_mean):
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
            "[Q-Tuning] Non-finite loss detected. "
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


def loss_function(args, batch, num_microbatches, logits):
    num_tokens = sum(batch["response_lengths"])
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
