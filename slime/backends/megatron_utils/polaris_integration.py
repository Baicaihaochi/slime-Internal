"""
Integration module for POLARIS features in Megatron actor.

This module provides functions to integrate POLARIS dynamic sampling
and reward tracking into the training loop.
"""

from pathlib import Path
import numbers
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.utils.polaris_utils import (
    DynamicSampleReplacer,
    RewardTracker,
    aggregate_rewards_per_prompt,
    extract_sample_indices,
    replace_samples_in_rollout,
)


def init_polaris_components(args):
    """
    Initialize POLARIS components (reward tracker and dynamic replacer).

    Args:
        args: Training arguments

    Returns:
        Tuple of (reward_tracker, dynamic_replacer)
    """
    # Initialize reward tracker
    if args.enable_polaris_reward_tracking:
        if args.polaris_reward_tracking_dir is None:
            # Default to data directory
            data_path = getattr(args, "rollout_data_path", None) or getattr(args, "prompt_data", None)
            if data_path:
                tracking_dir = str(Path(data_path).parent)
            else:
                tracking_dir = "polaris_tracking"
        else:
            tracking_dir = args.polaris_reward_tracking_dir

        experiment_name = (
            getattr(args, "wandb_name", None)
            or getattr(args, "wandb_group", None)
            or "experiment"
        )

        reward_tracker = RewardTracker(
            save_dir=tracking_dir,
            experiment_name=experiment_name,
            enabled=True,
        )
    else:
        reward_tracker = RewardTracker(
            save_dir="",
            experiment_name="",
            enabled=False,
        )

    # Initialize dynamic sample replacer
    if args.enable_polaris_dynamic_sampling:
        dynamic_replacer = DynamicSampleReplacer(
            enabled=True,
            good_reward_range=(args.polaris_good_reward_min, args.polaris_good_reward_max),
            min_good_ratio=args.polaris_min_good_ratio,
            verbose=args.polaris_verbose,
        )
    else:
        dynamic_replacer = DynamicSampleReplacer(
            enabled=False,
        )

    return reward_tracker, dynamic_replacer


def extract_rewards_from_rollout_data(
    rollout_data: Dict,
    n_samples_per_prompt: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and aggregate rewards from rollout data.

    Args:
        rollout_data: Dictionary containing rollout data
        n_samples_per_prompt: Number of samples per prompt (rollout_n)

    Returns:
        per_rollout_rewards: Rewards for each rollout, shape (batch_size * n_samples,)
        per_prompt_rewards: Average reward per prompt, shape (batch_size,)
    """
    # Prefer raw reward when shape matches local samples; otherwise fall back to rewards.
    rewards = None
    local_count = len(rollout_data.get("tokens", []))
    if "raw_reward" in rollout_data:
        rr = rollout_data["raw_reward"]
        if isinstance(rr, (list, tuple)) and len(rr) == local_count:
            rewards = rr
    if rewards is None and "rewards" in rollout_data:
        rewards = rollout_data["rewards"]

    if rewards is not None:
        if isinstance(rewards, list):
            first = rewards[0]
            if isinstance(first, torch.Tensor):
                per_rollout_rewards = np.array([
                    r.item() if r.numel() == 1 else r.sum().item() for r in rewards
                ])
            else:
                per_rollout_rewards = np.array(rewards, dtype=float)
        elif isinstance(rewards, torch.Tensor):
            per_rollout_rewards = rewards.detach().cpu().numpy()
        else:
            per_rollout_rewards = np.array(rewards, dtype=float)
    else:
        num_samples = len(rollout_data.get("tokens", []))
        per_rollout_rewards = np.zeros(num_samples)

    # Aggregate to per-prompt rewards
    per_prompt_rewards = aggregate_rewards_per_prompt(per_rollout_rewards, n_samples_per_prompt)

    return per_rollout_rewards, per_prompt_rewards


def apply_polaris_to_rollout_data(
    args,
    rollout_data: Dict,
    rollout_id: int,
    reward_tracker: RewardTracker,
    dynamic_replacer: DynamicSampleReplacer,
) -> Tuple[Dict, Dict]:
    """Apply POLARIS features to rollout data before training."""
    is_controller = mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage()

    polaris_stats: Dict = {}
    replacement_plan: Optional[Dict[str, List[int]]] = None

    if is_controller:
        n_samples_per_prompt = getattr(args, "n_samples_per_prompt", 1)
        per_rollout_rewards, per_prompt_rewards = extract_rewards_from_rollout_data(
            rollout_data,
            n_samples_per_prompt=n_samples_per_prompt,
        )

        polaris_stats["polaris/mean_reward"] = per_prompt_rewards.mean()
        polaris_stats["polaris/std_reward"] = per_prompt_rewards.std()
        polaris_stats["polaris/reward_0_count"] = (per_prompt_rewards == 0).sum()
        polaris_stats["polaris/reward_1_count"] = (per_prompt_rewards == 1).sum()
        polaris_stats["polaris/reward_mid_count"] = (
            (per_prompt_rewards > 0) & (per_prompt_rewards < 1)
        ).sum()

        # Only one writer across DP and CP: DP(rank==0, with_context_parallel=True) and CP(rank==0 if available)
        dp_rank_with_cp = mpu.get_data_parallel_rank(with_context_parallel=True)
        is_dp0_with_cp = dp_rank_with_cp == 0
        cp_rank_fn = getattr(mpu, "get_context_parallel_rank", None)
        current_cp_rank = cp_rank_fn() if cp_rank_fn is not None else None
        is_cp0 = True if cp_rank_fn is None else (current_cp_rank == 0)
        if reward_tracker.enabled:
            # Build per-prompt indices aligned with per_prompt_rewards
            num_samples_local = len(rollout_data.get("tokens", []))
            rollout_n = n_samples_per_prompt
            if (
                "sample_indices" in rollout_data
                and isinstance(rollout_data["sample_indices"], list)
                and len(rollout_data["sample_indices"]) == num_samples_local
                and rollout_n > 0
                and num_samples_local % rollout_n == 0
            ):
                # Take the first sample index of each prompt group
                per_prompt_indices = rollout_data["sample_indices"][::rollout_n]
            else:
                # Fallback to sequential indices
                per_prompt_indices = list(range(len(per_prompt_rewards)))

            tracker_payload = {
                "indices": [int(idx) for idx in per_prompt_indices],
                "scores": [float(score) for score in per_prompt_rewards.tolist()],
            }
            dp_src_with_cp = mpu.get_data_parallel_src_rank(with_context_parallel=True)
            dp_world_size_with_cp = mpu.get_data_parallel_world_size(with_context_parallel=True)
            gathered_tracker_payloads = (
                [None] * dp_world_size_with_cp if dp_rank_with_cp == dp_src_with_cp else None
            )

            dist.gather_object(
                tracker_payload,
                gathered_tracker_payloads,
                dst=dp_src_with_cp,
                group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
            )

            if is_dp0_with_cp and is_cp0 and dp_rank_with_cp == dp_src_with_cp:
                combined_indices: list[int] = []
                combined_scores: list[float] = []
                seen_scores: dict[int, list[float]] = {}
                for entry in gathered_tracker_payloads or []:
                    if not entry:
                        continue
                    entry_indices = entry.get("indices", [])
                    entry_scores = entry.get("scores", [])
                    for idx, score in zip(entry_indices, entry_scores):
                        score_history = seen_scores.setdefault(idx, [])
                        if any(math.isclose(score, logged, rel_tol=1e-6, abs_tol=1e-6) for logged in score_history):
                            continue
                        score_history.append(score)
                        combined_indices.append(idx)
                        combined_scores.append(score)

                reward_tracker.log_batch_rewards(
                    sample_indices=combined_indices,
                    rewards=np.array(combined_scores, dtype=float),
                    rollout_id=rollout_id,
                )
                tracker_stats = reward_tracker.get_statistics()
                polaris_stats["polaris/tracker_total_batches"] = tracker_stats["total_batches"]
                polaris_stats["polaris/tracker_total_samples"] = tracker_stats["total_samples"]

        if dynamic_replacer.enabled:
            (
                rollout_data,
                modified_per_prompt_rewards,
                replacement_stats,
                replacement_plan,
            ) = dynamic_replacer.replace_samples(
                rollout_data=rollout_data,
                rewards=per_prompt_rewards,
                rollout_n=n_samples_per_prompt,
            )

            polaris_stats.update({
                f"polaris/replacer_{k}": v for k, v in replacement_stats.items()
            })
            for bool_key in ("polaris/replacer_enabled", "polaris/replacer_replaced"):
                if bool_key in polaris_stats:
                    polaris_stats[bool_key] = 1.0 if polaris_stats[bool_key] else 0.0
            for bool_key in ("polaris/replacer_enabled", "polaris/replacer_replaced"):
                if bool_key in polaris_stats:
                    polaris_stats[bool_key] = 1.0 if polaris_stats[bool_key] else 0.0

            # Optionally skip this batch to align with verl when insufficient good samples
            if (
                not replacement_stats.get("replaced", False)
                and replacement_stats.get("reason") == "insufficient_good_samples"
                and getattr(args, "polaris_skip_batch_when_insufficient", False)
            ):
                # Mark a flag so the caller can choose to skip training on this batch.
                polaris_stats["polaris/skip_batch_due_to_insufficient_good"] = 1
                # No replacement plan should be applied across ranks in this case.
                replacement_plan = None

            if replacement_stats.get("replaced", False):
                polaris_stats["polaris/mean_reward_after"] = modified_per_prompt_rewards.mean()
                polaris_stats["polaris/std_reward_after"] = modified_per_prompt_rewards.std()

        replacer_stats = dynamic_replacer.get_statistics()
        polaris_stats["polaris/replacer_total_calls"] = replacer_stats["total_calls"]
        polaris_stats["polaris/replacer_total_replacements"] = replacer_stats["total_replacements"]
        polaris_stats["polaris/replacer_rate"] = replacer_stats["replacement_rate"]

    mp_group = mpu.get_model_parallel_group()
    if mp_group is not None and dist.get_world_size(group=mp_group) > 1:
        plan_buffer = [replacement_plan]
        dist.broadcast_object_list(plan_buffer, src=mpu.get_model_parallel_src_rank(), group=mp_group)
        replacement_plan = plan_buffer[0]

    if replacement_plan:
        rollout_data = replace_samples_in_rollout(
            rollout_data,
            replacement_plan["bad_indices"],
            replacement_plan["chosen_indices"],
        )

    return rollout_data, polaris_stats if is_controller else {}


def log_polaris_stats(rollout_id, args, polaris_stats):
    """
    Log POLARIS statistics to console and wandb.

    Args:
        rollout_id: Current rollout/step ID
        args: Training arguments
        polaris_stats: Dictionary of POLARIS statistics
    """
    payload = polaris_stats if polaris_stats else {"_polaris_empty": True}

    # Only log from main rank
    if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
        # Gather statistics across data parallel ranks if needed
        gathered_stats = [None] * mpu.get_data_parallel_world_size(with_context_parallel=True)
        dist.gather_object(
            payload,
            gathered_stats,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )

        if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
            # Average statistics while tolerating ranks without this key.
            valid_stats = [
                s for s in gathered_stats if isinstance(s, dict) and not s.get("_polaris_empty", False)
            ]
            if not valid_stats:
                return

            rank_count = len(valid_stats)

            dp_world_size_with_cp = mpu.get_data_parallel_world_size(with_context_parallel=True)
            dp_world_size_without_cp = mpu.get_data_parallel_world_size(with_context_parallel=False)

            cp_world_size_fn = getattr(mpu, "get_context_parallel_world_size", None)
            cp_world_size = cp_world_size_fn() if cp_world_size_fn is not None else 1

            expected_controller_count = dp_world_size_with_cp
            assert rank_count == expected_controller_count, (
                f"Missing POLARIS stats: expected {expected_controller_count} controller reports, "
                f"got {rank_count}. This may indicate a crash, early exit, or communication issue in one or more ranks."
            )

            rank_count = len(valid_stats)

            averaged_stats = {}
            all_keys = set().union(*(s.keys() for s in valid_stats))
            for key in all_keys:
                values = [s[key] for s in valid_stats if key in s]
                if not values:
                    continue

                if all(isinstance(v, numbers.Number) for v in values):
                    averaged_stats[key] = sum(values) / len(values)
                else:
                    averaged_stats[key] = values[0]

            averaged_stats["polaris/dp_world_size"] = dp_world_size_with_cp
            averaged_stats["polaris/dp_world_size_without_cp"] = dp_world_size_without_cp
            averaged_stats["polaris/cp_world_size"] = cp_world_size
            reward_bucket_keys = [
                "polaris/reward_0_count",
                "polaris/reward_mid_count",
                "polaris/reward_1_count",
            ]
            if all(key in averaged_stats for key in reward_bucket_keys):
                reward_0_avg = averaged_stats["polaris/reward_0_count"]
                reward_mid_avg = averaged_stats["polaris/reward_mid_count"]
                reward_1_avg = averaged_stats["polaris/reward_1_count"]
                batch_total_prompts = (reward_0_avg + reward_mid_avg + reward_1_avg) * dp_world_size_without_cp
                averaged_stats["polaris/batch_total_prompts"] = batch_total_prompts
                averaged_stats["polaris/batch_solve_none_total"] = reward_0_avg * dp_world_size_without_cp
                averaged_stats["polaris/batch_solve_partial_total"] = reward_mid_avg * dp_world_size_without_cp
                averaged_stats["polaris/batch_solve_all_total"] = reward_1_avg * dp_world_size_without_cp
            if "polaris/replacer_replaced" in averaged_stats:
                avg_replaced = averaged_stats["polaris/replacer_replaced"]
                successful_ranks = avg_replaced * rank_count
                averaged_stats["polaris/replacer_successful_ranks"] = successful_ranks
                averaged_stats["polaris/replacer_total_ranks"] = rank_count
                averaged_stats["polaris/replacer_success_rate"] = (
                    successful_ranks / rank_count if rank_count > 0 else 0.0
                )
            print(f"POLARIS stats {rollout_id}: {averaged_stats}")

            if args.use_wandb:
                import wandb
                averaged_stats["rollout/step"] = (
                    rollout_id
                    if not args.wandb_always_use_train_step
                    else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
                )
                wandb.log(averaged_stats)
    else:
        dist.gather_object(
            payload,
            None,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )