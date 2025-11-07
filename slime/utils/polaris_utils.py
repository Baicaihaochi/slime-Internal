"""
POLARIS utilities for dynamic sampling and reward tracking.

This module implements the key tricks from POLARIS:
1. Reward tracking - Save reward history to JSONL for difficulty filtering
2. Dynamic sample replacement - Replace trivial samples (reward=0 or 1) with medium-difficulty ones
"""

import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _clone_sample_entry(entry):
    """Clone a rollout entry so replacing one sample does not alias another."""
    if isinstance(entry, torch.Tensor):
        return entry.clone()
    try:
        return deepcopy(entry)
    except Exception:
        return entry


def replace_samples_in_rollout(
    rollout_data: Dict,
    bad_indices: List[int],
    chosen_indices: List[int],
) -> Dict:
    """Apply a replacement plan to rollout data and return a modified copy."""
    if not bad_indices:
        return rollout_data

    num_samples = len(rollout_data.get("tokens", []))
    if num_samples == 0:
        return rollout_data

    modified = {}
    for key, value in rollout_data.items():
        if isinstance(value, list) and len(value) == num_samples:
            updated_list = list(value)
            for bad_idx, chosen_idx in zip(bad_indices, chosen_indices):
                updated_list[bad_idx] = _clone_sample_entry(updated_list[chosen_idx])
            modified[key] = updated_list
        else:
            modified[key] = value
    return modified


class RewardTracker:
    """
    Track and save reward scores for each sample during training.

    This enables post-training analysis and difficulty-based data filtering,
    similar to POLARIS's drop_easy_data.py functionality.
    """

    def __init__(
        self,
        save_dir: str,
        experiment_name: str,
        enabled: bool = True,
    ):
        """
        Args:
            save_dir: Directory to save reward tracking files
            experiment_name: Name of the experiment (used as filename)
            enabled: Whether to enable reward tracking
        """
        self.enabled = enabled
        if not self.enabled:
            return

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / f"{experiment_name}.jsonl"

        # Track statistics
        self.total_batches = 0
        self.total_samples = 0

    def log_batch_rewards(
        self,
        sample_indices: List[int],
        rewards: np.ndarray,
        rollout_id: Optional[int] = None,
    ):
        """
        Log rewards for a batch of samples.

        Args:
            sample_indices: List of sample indices in the dataset
            rewards: Array of reward scores, shape (batch_size,)
            rollout_id: Optional rollout/step ID
        """
        if not self.enabled:
            return

        # Convert to list if needed
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(rewards, np.ndarray):
            rewards = rewards.tolist()

        # Create log entry
        log_entry = {
            "index": sample_indices,
            "score": rewards,
        }
        if rollout_id is not None:
            log_entry["rollout_id"] = rollout_id

        # Append to JSONL file
        with open(self.save_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        self.total_batches += 1
        self.total_samples += len(sample_indices)

    def get_statistics(self) -> Dict[str, int]:
        """Get tracking statistics."""
        return {
            "total_batches": self.total_batches,
            "total_samples": self.total_samples,
            "save_path": str(self.save_path),
        }


class DynamicSampleReplacer:
    """
    Dynamically replace trivial samples (reward=0 or 1) with medium-difficulty ones.

    This implements POLARIS's dynamic sampling trick to maintain training data quality
    and avoid wasting compute on trivial samples.
    """

    def __init__(
        self,
        enabled: bool = True,
        good_reward_range: Tuple[float, float] = (0.0, 1.0),
        min_good_ratio: float = 0.33,
        verbose: bool = True,
    ):
        """
        Args:
            enabled: Whether to enable dynamic sample replacement
            good_reward_range: (min, max) range for "good" samples (exclusive)
            min_good_ratio: Minimum ratio of good samples required to perform replacement
            verbose: Whether to print replacement information
        """
        self.enabled = enabled
        self.good_reward_range = good_reward_range
        self.min_good_ratio = min_good_ratio
        self.verbose = verbose

        # Statistics
        self.total_calls = 0
        self.total_replacements = 0
        self.total_skipped = 0

    def should_replace_batch(
        self,
        rewards: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        """
        Determine if batch should undergo sample replacement.

        Args:
            rewards: Array of reward scores, shape (batch_size,)

        Returns:
            should_replace: Whether replacement should be performed
            good_mask: Boolean mask indicating "good" samples
        """
        if not self.enabled:
            return False, np.ones(len(rewards), dtype=bool)

        # Identify "good" samples (not trivial)
        good_mask = (rewards > self.good_reward_range[0]) & (rewards < self.good_reward_range[1])

        good_count = good_mask.sum()
        total_count = len(rewards)
        good_ratio = good_count / total_count if total_count > 0 else 0

        # Only replace if we have enough good samples
        should_replace = good_ratio > self.min_good_ratio

        return should_replace, good_mask

    def get_replacement_indices(
        self,
        good_mask: np.ndarray,
        rollout_n: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get indices for sample replacement.

        Args:
            good_mask: Boolean mask indicating "good" samples (per prompt)
            rollout_n: Number of rollouts per prompt

        Returns:
            bad_indices: Indices of samples to be replaced (expanded for all rollouts)
            chosen_indices: Indices of good samples to use as replacements
        """
        # Get bad and good indices at the prompt level
        bad_indices_prompt = np.where(~good_mask)[0]
        good_indices_prompt = np.where(good_mask)[0]

        num_bad = len(bad_indices_prompt)
        num_good = len(good_indices_prompt)

        if num_bad == 0 or num_good == 0:
            return np.array([]), np.array([])

        # Sample with replacement if necessary
        if num_good >= num_bad:
            chosen_prompt_indices = np.random.choice(good_indices_prompt, size=num_bad, replace=False)
        else:
            chosen_prompt_indices = np.random.choice(good_indices_prompt, size=num_bad, replace=True)

        # Expand to all rollouts
        bad_indices = self._expand_to_rollouts(bad_indices_prompt, rollout_n)
        chosen_indices = self._expand_to_rollouts(chosen_prompt_indices, rollout_n)

        return bad_indices, chosen_indices

    @staticmethod
    def _expand_to_rollouts(indices: np.ndarray, rollout_n: int) -> np.ndarray:
        """
        Expand prompt-level indices to include all rollouts.

        For example, if indices=[0, 2] and rollout_n=3:
        - Prompt 0 has rollouts at positions [0, 1, 2]
        - Prompt 2 has rollouts at positions [6, 7, 8]
        - Returns: [0, 1, 2, 6, 7, 8]
        """
        expanded = []
        for idx in indices:
            start = idx * rollout_n
            expanded.extend(range(start, start + rollout_n))
        return np.array(expanded)

    def replace_samples(
        self,
        rollout_data: Dict,
        rewards: np.ndarray,
        rollout_n: int = 1,
    ) -> Tuple[Dict, np.ndarray, Dict[str, int], Optional[Dict[str, List[int]]]]:
        """
        Replace trivial samples in rollout data.

        Args:
            rollout_data: Dictionary containing rollout data with per-sample fields.
            rewards: Array of average rewards per prompt, shape (batch_size,)
            rollout_n: Number of rollouts per prompt

        Returns:
            modified_rollout_data: Rollout data with bad samples replaced
            modified_rewards: Updated reward array after replacement
            stats: Dictionary of replacement statistics
            replacement_plan: Replacement indices to replay on other ranks
        """
        self.total_calls += 1

        if not self.enabled:
            return rollout_data, rewards, {"enabled": False}, None

        # Check if replacement should be performed
        should_replace, good_mask = self.should_replace_batch(rewards)

        if not should_replace:
            self.total_skipped += 1
            if self.verbose:
                print("=" * 60)
                print("[POLARIS Dynamic Sampling] Warning: Skipping replacement")
                print(f"  Reason: Insufficient good samples ({good_mask.sum()}/{len(good_mask)}, "
                      f"ratio={good_mask.sum()/len(good_mask):.2%} < {self.min_good_ratio:.2%})")
                print(f"  Most samples have trivial rewards (0 or 1)")
                print(f"  Check your data difficulty distribution!")
                print("=" * 60)
            return rollout_data, rewards, {
                "enabled": True,
                "replaced": False,
                "reason": "insufficient_good_samples",
                "good_count": int(good_mask.sum()),
                "total_count": len(good_mask),
            }, None

        # Get replacement indices
        bad_indices, chosen_indices = self.get_replacement_indices(good_mask, rollout_n)

        if len(bad_indices) == 0:
            return rollout_data, rewards, {"enabled": True, "replaced": False, "reason": "no_bad_samples"}, None

        num_samples_total = len(rollout_data.get("tokens", []))
        valid_mask = (bad_indices < num_samples_total) & (chosen_indices < num_samples_total)
        if not valid_mask.all():
            if self.verbose:
                invalid = int((~valid_mask).sum())
                print(f"[POLARIS Dynamic Sampling] Warning: discard {invalid} invalid replacement pairs")
            bad_indices = bad_indices[valid_mask]
            chosen_indices = chosen_indices[valid_mask]

        if len(bad_indices) == 0:
            return rollout_data, rewards, {"enabled": True, "replaced": False, "reason": "invalid_indices"}, None

        # Apply replacements and recompute rewards
        modified_data = replace_samples_in_rollout(rollout_data, bad_indices.tolist(), chosen_indices.tolist())

        if isinstance(rewards, torch.Tensor):
            modified_rewards = rewards.clone()
        else:
            modified_rewards = rewards.copy()

        reward_key = "raw_reward" if "raw_reward" in modified_data else "rewards"
        reward_list = modified_data.get(reward_key)
        if isinstance(reward_list, torch.Tensor):
            reward_array = reward_list.detach().cpu().numpy()
        elif isinstance(reward_list, (list, np.ndarray)):
            reward_array = np.array([
                r if isinstance(r, (int, float)) else r.item() if hasattr(r, "item") else r
                for r in reward_list
            ], dtype=float)
        else:
            reward_array = None

        if reward_array is not None and reward_array.size >= rollout_n and reward_array.size % rollout_n == 0:
            modified_rewards = reward_array.reshape(-1, rollout_n).mean(axis=1)
        else:
            modified_rewards = np.array(modified_rewards, dtype=float)

        self.total_replacements += 1

        if self.verbose:
            print("=" * 60)
            print("[POLARIS Dynamic Sampling] Sample Replacement Performed")
            print(f"  Before: {rewards.tolist()}")
            print(f"  After:  {modified_rewards.tolist()}")
            print(f"  Replaced {len(bad_indices)} samples ({len(bad_indices)//rollout_n} prompts × {rollout_n} rollouts)")
            print("=" * 60)

        stats = {
            "enabled": True,
            "replaced": True,
            "num_bad_samples": len(bad_indices),
            "num_bad_prompts": len(bad_indices) // rollout_n,
            "good_count": int(good_mask.sum()),
            "total_count": len(good_mask),
        }

        plan = {
            "bad_indices": bad_indices.tolist(),
            "chosen_indices": chosen_indices.tolist(),
            "rollout_n": rollout_n,
        }

        return modified_data, modified_rewards, stats, plan

    def get_statistics(self) -> Dict[str, int]:
        """Get replacement statistics."""
        return {
            "total_calls": self.total_calls,
            "total_replacements": self.total_replacements,
            "total_skipped": self.total_skipped,
            "replacement_rate": self.total_replacements / self.total_calls if self.total_calls > 0 else 0,
        }


def aggregate_rewards_per_prompt(
    rewards: np.ndarray,
    rollout_n: int,
) -> np.ndarray:
    """
    Aggregate per-rollout rewards to per-prompt average rewards.

    Args:
        rewards: Array of rewards, shape (batch_size * rollout_n,)
        rollout_n: Number of rollouts per prompt

    Returns:
        avg_rewards: Average reward per prompt, shape (batch_size,)
    """
    return rewards.reshape(-1, rollout_n).mean(axis=1)


def extract_sample_indices(
    rollout_data: Dict,
    index_key: str = "index",
) -> List[int]:
    """
    Extract sample indices from rollout data.

    Args:
        rollout_data: Dictionary containing rollout data
        index_key: Key for sample indices in metadata

    Returns:
        sample_indices: List of sample indices
    """

    if "sample_indices" in rollout_data and isinstance(rollout_data["sample_indices"], list):
        return rollout_data["sample_indices"]

    # Try to get from metadata
    if "metadata" in rollout_data:
        metadata = rollout_data["metadata"]
        if isinstance(metadata, list):
            indices = []
            for meta in metadata:
                if isinstance(meta, dict) and index_key in meta:
                    indices.append(meta[index_key])
                else:
                    indices.append(-1)  # Unknown index
            return indices

    # If not available, use sequential indices
    batch_size = len(rollout_data.get("tokens", []))
    return list(range(batch_size))
