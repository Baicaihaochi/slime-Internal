#!/usr/bin/env python
"""
Filter easy data based on reward tracking logs (POLARIS-style).

This script reads the reward tracking JSONL files generated during training
and filters out samples with high average rewards (easy samples) from the dataset.

Usage:
    python polaris_filter_easy_data.py \
        --data-path data/train.json \
        --reward-log polaris_tracking/experiment.jsonl \
        --output data/train_filtered.json \
        --threshold 0.9
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def process_reward_log(jsonl_path: str, threshold: float = 0.9) -> set:
    """
    Process reward tracking JSONL to identify easy samples.

    Args:
        jsonl_path: Path to the reward tracking JSONL file
        threshold: Average reward threshold above which samples are considered "easy"

    Returns:
        Set of sample indices to remove
    """
    index_to_scores = defaultdict(list)

    # Read all reward entries
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            indices = entry['index']
            scores = entry['score']

            for idx, score in zip(indices, scores):
                index_to_scores[idx].append(score)

    # Compute average and filter
    remove_indices = set()
    for idx, scores in index_to_scores.items():
        avg_score = sum(scores) / len(scores)
        if avg_score > threshold:
            remove_indices.add(idx)

    print(f"Total unique samples: {len(index_to_scores)}")
    print(f"Samples to remove (avg reward > {threshold}): {len(remove_indices)}")
    print(f"Remaining samples: {len(index_to_scores) - len(remove_indices)}")

    return remove_indices


def filter_json_data(input_path: str, output_path: str, remove_indices: set):
    """
    Filter JSON data by removing specified indices.

    Supports both columnar format (like your example) and list format.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        # List format: [{"problem": ..., "conversations": ...}, ...]
        filtered_data = [item for i, item in enumerate(data) if i not in remove_indices]
    elif isinstance(data, dict):
        # Columnar format: {"problem": {"0": ..., "1": ...}, ...}
        first_key = list(data.keys())[0]
        if isinstance(data[first_key], dict):
            # Get all indices
            all_indices = set(data[first_key].keys())
            keep_indices = sorted([idx for idx in all_indices if int(idx) not in remove_indices])

            # Rebuild columnar data with only kept indices
            filtered_data = {}
            for key in data.keys():
                filtered_data[key] = {}
                for new_idx, old_idx in enumerate(keep_indices):
                    filtered_data[key][str(new_idx)] = data[key][old_idx]
        else:
            raise ValueError("Unexpected JSON structure")
    else:
        raise ValueError("Unsupported JSON format")

    # Save filtered data
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"Filtered data saved to: {output_path}")


def filter_jsonl_data(input_path: str, output_path: str, remove_indices: set):
    """Filter JSONL data by removing specified indices."""
    with open(output_path, 'w') as out_f:
        with open(input_path, 'r') as in_f:
            for i, line in enumerate(in_f):
                if i not in remove_indices:
                    out_f.write(line)

    print(f"Filtered data saved to: {output_path}")


def filter_parquet_data(input_path: str, output_path: str, remove_indices: set):
    """Filter Parquet data by removing specified indices."""
    df = pd.read_parquet(input_path)
    print(f"Original dataframe size: {len(df)}")

    # Assume dataframe has an implicit index
    mask = ~df.index.isin(remove_indices)
    filtered_df = df[mask].reset_index(drop=True)

    print(f"Filtered dataframe size: {len(filtered_df)}")
    filtered_df.to_parquet(output_path)
    print(f"Filtered data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Filter easy data based on reward tracking")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input data file (.json, .jsonl, or .parquet)",
    )
    parser.add_argument(
        "--reward-log",
        type=str,
        required=True,
        help="Path to reward tracking JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output filtered data file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Reward threshold for filtering (default: 0.9)",
    )

    args = parser.parse_args()

    # Process reward log to get indices to remove
    print(f"Processing reward log: {args.reward_log}")
    remove_indices = process_reward_log(args.reward_log, args.threshold)

    # Filter data based on file format
    input_path = Path(args.data_path)
    print(f"\nFiltering data: {args.data_path}")

    if input_path.suffix == '.json':
        filter_json_data(args.data_path, args.output, remove_indices)
    elif input_path.suffix == '.jsonl':
        filter_jsonl_data(args.data_path, args.output, remove_indices)
    elif input_path.suffix == '.parquet':
        filter_parquet_data(args.data_path, args.output, remove_indices)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    print("\nFiltering complete!")


if __name__ == "__main__":
    main()
