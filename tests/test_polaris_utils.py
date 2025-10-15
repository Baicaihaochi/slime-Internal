"""
Unit tests for POLARIS utilities.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from slime.utils.polaris_utils import (
    DynamicSampleReplacer,
    RewardTracker,
    aggregate_rewards_per_prompt,
    extract_sample_indices,
)


class TestRewardTracker:
    """Test RewardTracker functionality."""

    def test_init_enabled(self):
        """Test initialization with tracking enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RewardTracker(
                save_dir=tmpdir,
                experiment_name="test_exp",
                enabled=True,
            )
            assert tracker.enabled
            assert tracker.save_path == Path(tmpdir) / "test_exp.jsonl"

    def test_init_disabled(self):
        """Test initialization with tracking disabled."""
        tracker = RewardTracker(
            save_dir="",
            experiment_name="",
            enabled=False,
        )
        assert not tracker.enabled

    def test_log_batch_rewards(self):
        """Test logging batch rewards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RewardTracker(
                save_dir=tmpdir,
                experiment_name="test",
                enabled=True,
            )

            indices = [0, 1, 2, 3]
            rewards = np.array([0.5, 0.8, 0.0, 1.0])

            tracker.log_batch_rewards(indices, rewards, rollout_id=0)

            # Verify file was created and contains correct data
            assert tracker.save_path.exists()

            with open(tracker.save_path, 'r') as f:
                line = f.readline()
                entry = json.loads(line)

            assert entry["index"] == indices
            assert entry["score"] == rewards.tolist()
            assert entry["rollout_id"] == 0

    def test_log_batch_rewards_disabled(self):
        """Test that logging does nothing when disabled."""
        tracker = RewardTracker(
            save_dir="",
            experiment_name="",
            enabled=False,
        )

        # Should not raise error
        tracker.log_batch_rewards([0, 1], np.array([0.5, 0.5]))

    def test_statistics(self):
        """Test statistics tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RewardTracker(tmpdir, "test", enabled=True)

            tracker.log_batch_rewards([0, 1], np.array([0.5, 0.5]))
            tracker.log_batch_rewards([2, 3, 4], np.array([0.7, 0.3, 0.9]))

            stats = tracker.get_statistics()
            assert stats["total_batches"] == 2
            assert stats["total_samples"] == 5


class TestDynamicSampleReplacer:
    """Test DynamicSampleReplacer functionality."""

    def test_init(self):
        """Test initialization."""
        replacer = DynamicSampleReplacer(
            enabled=True,
            good_reward_range=(0.0, 1.0),
            min_good_ratio=0.33,
        )
        assert replacer.enabled
        assert replacer.good_reward_range == (0.0, 1.0)
        assert replacer.min_good_ratio == 0.33

    def test_should_replace_batch_success(self):
        """Test successful replacement decision."""
        replacer = DynamicSampleReplacer(enabled=True, min_good_ratio=0.3)

        rewards = np.array([0.0, 0.5, 0.7, 1.0, 0.3, 0.8])
        should_replace, good_mask = replacer.should_replace_batch(rewards)

        assert should_replace
        assert good_mask.sum() == 4  # 0.5, 0.7, 0.3, 0.8

    def test_should_replace_batch_insufficient(self):
        """Test replacement decision with insufficient good samples."""
        replacer = DynamicSampleReplacer(enabled=True, min_good_ratio=0.5)

        rewards = np.array([0.0, 0.5, 1.0, 1.0, 0.0, 0.0])  # Only 1/6 good
        should_replace, good_mask = replacer.should_replace_batch(rewards)

        assert not should_replace
        assert good_mask.sum() == 1

    def test_get_replacement_indices(self):
        """Test getting replacement indices."""
        replacer = DynamicSampleReplacer(enabled=True)

        good_mask = np.array([False, True, True, False, True, False])
        rollout_n = 2

        bad_indices, chosen_indices = replacer.get_replacement_indices(good_mask, rollout_n)

        # Should have 3 bad prompts * 2 rollouts = 6 indices
        assert len(bad_indices) == 6
        assert len(chosen_indices) == 6

        # Verify expansion
        # Bad prompts are 0, 3, 5 -> rollouts [0,1], [6,7], [10,11]
        expected_bad = [0, 1, 6, 7, 10, 11]
        assert sorted(bad_indices.tolist()) == expected_bad

    def test_replace_samples(self):
        """Test full sample replacement."""
        replacer = DynamicSampleReplacer(enabled=True, min_good_ratio=0.3, verbose=False)

        # Create mock rollout data
        rollout_data = {
            "tokens": [
                torch.tensor([1, 2, 3]),  # rollout 0 of prompt 0
                torch.tensor([4, 5, 6]),  # rollout 1 of prompt 0
                torch.tensor([7, 8, 9]),  # rollout 0 of prompt 1
                torch.tensor([10, 11, 12]),  # rollout 1 of prompt 1
            ],
            "rewards": [0.0, 0.0, 0.5, 0.5],  # Prompt 0: bad (0), Prompt 1: good (0.5)
        }

        rewards = np.array([0.0, 0.5])  # Per-prompt rewards
        rollout_n = 2

        modified_data, modified_rewards, stats = replacer.replace_samples(
            rollout_data, rewards, rollout_n
        )

        assert stats["replaced"]
        assert stats["num_bad_prompts"] == 1

        # Verify tokens were replaced (prompt 0's rollouts should match prompt 1's)
        assert torch.equal(modified_data["tokens"][0], torch.tensor([7, 8, 9]))
        assert torch.equal(modified_data["tokens"][1], torch.tensor([10, 11, 12]))

    def test_replace_samples_skip(self):
        """Test skipping replacement when insufficient good samples."""
        replacer = DynamicSampleReplacer(enabled=True, min_good_ratio=0.8, verbose=False)

        rollout_data = {"tokens": [torch.tensor([1, 2, 3])]}
        rewards = np.array([0.0, 1.0])  # All bad

        modified_data, modified_rewards, stats = replacer.replace_samples(
            rollout_data, rewards, rollout_n=1
        )

        assert not stats["replaced"]
        assert stats["reason"] == "insufficient_good_samples"

    def test_statistics(self):
        """Test statistics tracking."""
        replacer = DynamicSampleReplacer(enabled=True, verbose=False)

        rollout_data = {"tokens": [torch.tensor([i]) for i in range(4)], "rewards": [0.0] * 4}

        # First call - should skip (all bad)
        replacer.replace_samples(rollout_data, np.array([0.0, 0.0]), rollout_n=2)

        # Second call - should replace
        replacer.replace_samples(rollout_data, np.array([0.0, 0.5]), rollout_n=2)

        stats = replacer.get_statistics()
        assert stats["total_calls"] == 2
        assert stats["total_replacements"] == 1
        assert stats["replacement_rate"] == 0.5


class TestHelperFunctions:
    """Test helper functions."""

    def test_aggregate_rewards_per_prompt(self):
        """Test reward aggregation."""
        rewards = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        rollout_n = 2

        avg_rewards = aggregate_rewards_per_prompt(rewards, rollout_n)

        expected = np.array([0.15, 0.35, 0.55])  # (0.1+0.2)/2, (0.3+0.4)/2, (0.5+0.6)/2
        np.testing.assert_array_almost_equal(avg_rewards, expected)

    def test_extract_sample_indices_from_metadata(self):
        """Test extracting indices from metadata."""
        rollout_data = {
            "metadata": [
                {"index": 10, "other": "data"},
                {"index": 20},
                {"index": 30},
            ],
            "tokens": [None, None, None],
        }

        indices = extract_sample_indices(rollout_data)
        assert indices == [10, 20, 30]

    def test_extract_sample_indices_default(self):
        """Test default index extraction when no metadata."""
        rollout_data = {
            "tokens": [None, None, None],
        }

        indices = extract_sample_indices(rollout_data)
        assert indices == [0, 1, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
