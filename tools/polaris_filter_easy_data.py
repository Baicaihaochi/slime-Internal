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
from numbers import Integral

import pandas as pd


def _normalize_index(idx):
    """Convert raw index values from logs/data into comparable ints."""
    if isinstance(idx, Integral):
        return int(idx)
    if isinstance(idx, str):
        idx = idx.strip()
        if not idx:
            return None
        try:
            return int(idx)
        except ValueError:
            return None
    return None


def _should_remove(idx, remove_indices, logged_indices):
    """Decide whether a sample should be removed based on log coverage."""
    if idx not in remove_indices:
        return False
    if not logged_indices:
        return True
    return idx in logged_indices


def _extract_index_from_record(record, fallback):
    """Pull a stable sample index from a record if present."""
    if isinstance(record, dict):
        candidate_dicts = [record]
        meta = record.get("meta")
        if isinstance(meta, dict):
            candidate_dicts.append(meta)

        for candidate in candidate_dicts:
            for key in ("original_index", "index", "id"):
                if key in candidate:
                    idx = _normalize_index(candidate[key])
                    if idx is not None:
                        return idx
    return fallback


def process_reward_log(jsonl_path: str, threshold: float = 0.9):
    """
    Process reward tracking JSONL to identify easy samples.

    Args:
        jsonl_path: Path to the reward tracking JSONL file
        threshold: Average reward threshold above which samples are considered "easy"

    Returns:
        (remove_indices, logged_indices)
    """
    index_to_scores = defaultdict(list)
    skipped_indices = 0

    # Read all reward entries
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            indices = entry['index']
            scores = entry['score']

            for raw_idx, score in zip(indices, scores):
                idx = _normalize_index(raw_idx)
                if idx is None:
                    skipped_indices += 1
                    continue
                index_to_scores[idx].append(score)

    # Compute average and filter
    remove_indices = set()
    for idx, scores in index_to_scores.items():
        avg_score = sum(scores) / len(scores)
        if avg_score > threshold:
            remove_indices.add(idx)

    logged_indices = set(index_to_scores.keys())

    print(f"Total unique samples in reward log: {len(logged_indices)}")
    print(f"Samples to remove (avg reward > {threshold}): {len(remove_indices)}")
    print(f"Samples retained within reward log coverage: {len(logged_indices) - len(remove_indices)}")
    if skipped_indices:
        print(f"Skipped {skipped_indices} log entries with non-integer indices.")

    return remove_indices, logged_indices


def filter_json_data(input_path: str, output_path: str, remove_indices: set, logged_indices: set):
    """
    Filter JSON data by removing specified indices.

    Supports both columnar format (like your example) and list format.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    removed_count = 0
    missing_in_log = 0
    unknown_index_count = 0
    filtered_len = None

    if isinstance(data, list):
        # List format: [{"problem": ..., "conversations": ...}, ...]
        filtered_data = []
        for i, item in enumerate(data):
            idx = _extract_index_from_record(item, i)
            if logged_indices and idx not in logged_indices:
                missing_in_log += 1
            if _should_remove(idx, remove_indices, logged_indices):
                removed_count += 1
                continue
            filtered_data.append(item)
        filtered_len = len(filtered_data)
    elif isinstance(data, dict):
        # Columnar format: {"problem": {"0": ..., "1": ...}, ...}
        first_key = next(iter(data.keys()))
        if isinstance(data[first_key], dict):
            keys = list(data.keys())
            filtered_data = {key: {} for key in keys}
            new_index = 0
            for old_idx in data[first_key].keys():
                idx_int = _normalize_index(old_idx)
                if idx_int is None:
                    unknown_index_count += 1
                else:
                    if logged_indices and idx_int not in logged_indices:
                        missing_in_log += 1
                    if _should_remove(idx_int, remove_indices, logged_indices):
                        removed_count += 1
                        continue
                for key in keys:
                    filtered_data[key][str(new_index)] = data[key][old_idx]
                new_index += 1
            filtered_len = new_index
        else:
            raise ValueError("Unexpected JSON structure")
    else:
        raise ValueError("Unsupported JSON format")

    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"Removed samples: {removed_count}")
    if logged_indices:
        print(f"Samples without reward log entries kept: {missing_in_log}")
    if unknown_index_count:
        print(f"Entries with non-integer indices kept: {unknown_index_count}")
    if filtered_len is not None:
        print(f"Filtered dataset size: {filtered_len}")
    print(f"Filtered data saved to: {output_path}")


def filter_jsonl_data(input_path: str, output_path: str, remove_indices: set, logged_indices: set):
    """Filter JSONL data by removing specified indices."""
    removed_count = 0
    missing_in_log = 0
    json_parse_failures = 0
    total_records = 0

    with open(output_path, 'w') as out_f:
        with open(input_path, 'r') as in_f:
            for i, line in enumerate(in_f):
                total_records += 1
                idx = i
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = None
                    json_parse_failures += 1
                if record is not None:
                    idx = _extract_index_from_record(record, i)
                if logged_indices and idx not in logged_indices:
                    missing_in_log += 1
                if _should_remove(idx, remove_indices, logged_indices):
                    removed_count += 1
                    continue
                out_f.write(line)

    print(f"Removed samples: {removed_count}")
    if logged_indices:
        print(f"Samples without reward log entries kept: {missing_in_log}")
    if json_parse_failures:
        print(f"Warning: {json_parse_failures} lines could not be parsed as JSON; fell back to line numbers.")
    print(f"Filtered dataset size: {total_records - removed_count}")
    print(f"Filtered data saved to: {output_path}")


def filter_parquet_data(input_path: str, output_path: str, remove_indices: set, logged_indices: set):
    """Filter Parquet data by removing specified indices."""
    df = pd.read_parquet(input_path)
    print(f"Original dataframe size: {len(df)}")

    removed_count = 0
    missing_in_log = 0
    unknown_index_count = 0
    removal_flags = []

    for idx_val in df.index:
        idx_int = _normalize_index(idx_val)
        if idx_int is None:
            unknown_index_count += 1
            removal_flags.append(False)
            continue
        if logged_indices and idx_int not in logged_indices:
            missing_in_log += 1
        remove_flag = _should_remove(idx_int, remove_indices, logged_indices)
        if remove_flag:
            removed_count += 1
        removal_flags.append(remove_flag)

    mask = ~pd.Series(removal_flags, index=df.index, dtype=bool)
    filtered_df = df[mask].reset_index(drop=True)

    print(f"Filtered dataframe size: {len(filtered_df)}")
    print(f"Removed samples: {removed_count}")
    if logged_indices:
        print(f"Samples without reward log entries kept: {missing_in_log}")
    if unknown_index_count:
        print(f"Entries with non-integer indices kept: {unknown_index_count}")
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
    remove_indices, logged_indices = process_reward_log(args.reward_log, args.threshold)

    # Filter data based on file format
    input_path = Path(args.data_path)
    print(f"\nFiltering data: {args.data_path}")

    if input_path.suffix == '.json':
        filter_json_data(args.data_path, args.output, remove_indices, logged_indices)
    elif input_path.suffix == '.jsonl':
        filter_jsonl_data(args.data_path, args.output, remove_indices, logged_indices)
    elif input_path.suffix == '.parquet':
        filter_parquet_data(args.data_path, args.output, remove_indices, logged_indices)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    print("\nFiltering complete!")


if __name__ == "__main__":
    main()
