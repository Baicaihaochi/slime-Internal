#!/usr/bin/env python
"""
Extract "hard" dataset samples based on POLARIS reward tracking logs.

This script scans a reward tracking JSONL file and keeps the entries whose
average reward falls below a user-provided threshold. The matching samples
are written to a new dataset file, preserving the original format
(.json, .jsonl, or .parquet).

Each retained sample in the output carries an `_original_index` field so that
you can trace it back to the exact line/row in the source dataset.

Example:
    python polaris_extract_hard_samples.py \
        --data-path data/train.jsonl \
        --reward-log polaris_tracking/run.jsonl \
        --output data/train_hard_cases.jsonl \
        --threshold 0.1 \
        --threshold-low 0.0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ORIGINAL_INDEX_KEY = "_original_index"


def collect_low_reward_indices(jsonl_path: str, threshold_high: float, threshold_low: float) -> set[int]:
    """
    Parse reward tracking logs and return indices whose average reward is within the thresholds.
    """
    index_to_scores: dict[int, list[float]] = defaultdict(list)

    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            indices = entry.get("index") or entry.get("indices")
            scores = entry.get("score") or entry.get("scores")
            if indices is None or scores is None:
                raise KeyError("Reward log entries must contain 'index'/'indices' and 'score'/'scores'.")

            for idx, score in zip(indices, scores):
                index_to_scores[int(idx)].append(float(score))

    keep_indices: set[int] = set()
    for idx, scores in index_to_scores.items():
        avg_score = sum(scores) / len(scores)
        if threshold_low <= avg_score < threshold_high:
            keep_indices.add(idx)

    print(f"Total tracked samples: {len(index_to_scores)}")
    print(
        "Samples kept "
        f"(avg reward in [{threshold_low}, {threshold_high}))"
        f": {len(keep_indices)}"
    )
    return keep_indices


def keep_from_json(input_path: str, output_path: str, keep_indices: set[int]) -> None:
    """
    Save only the selected indices from a JSON dataset.
    Supports list-of-objects and columnar dict formats.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        filtered = []
        for idx, item in enumerate(data):
            if idx not in keep_indices:
                continue
            if isinstance(item, dict):
                sample = dict(item)
                sample[ORIGINAL_INDEX_KEY] = idx
            else:
                sample = { "value": item, ORIGINAL_INDEX_KEY: idx }
            filtered.append(sample)
    elif isinstance(data, dict):
        first_key = next(iter(data))
        if isinstance(data[first_key], dict):
            available_indices = set(data[first_key].keys())
            selected_old_indices = [str(idx) for idx in sorted(keep_indices) if str(idx) in available_indices]

            filtered = {key: {} for key in data}
            filtered[ORIGINAL_INDEX_KEY] = {}
            for new_idx, old_idx in enumerate(selected_old_indices):
                for key in data:
                    filtered[key][str(new_idx)] = data[key][old_idx]
                filtered[ORIGINAL_INDEX_KEY][str(new_idx)] = int(old_idx)
        else:
            raise ValueError("Unsupported JSON column format.")
    else:
        raise ValueError("Unsupported JSON structure.")

    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    saved_count = len(filtered) if isinstance(filtered, list) else len(selected_old_indices)
    print(f"Saved {saved_count} samples to {output_path}")


def keep_from_jsonl(input_path: str, output_path: str, keep_indices: set[int]) -> None:
    """Write only the selected lines from a JSONL dataset."""
    kept = 0
    with open(output_path, "w") as out_f:
        with open(input_path, "r") as in_f:
            for idx, line in enumerate(in_f):
                if idx in keep_indices:
                    entry = json.loads(line)
                    if isinstance(entry, dict):
                        payload = dict(entry)
                        payload[ORIGINAL_INDEX_KEY] = idx
                    else:
                        payload = {"value": entry, ORIGINAL_INDEX_KEY: idx}
                    out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    kept += 1
    print(f"Saved {kept} samples to {output_path}")


def keep_from_parquet(input_path: str, output_path: str, keep_indices: set[int]) -> None:
    """Write only the selected rows from a Parquet dataset."""
    df = pd.read_parquet(input_path)
    filtered_df = df[df.index.isin(keep_indices)].copy()
    filtered_df[ORIGINAL_INDEX_KEY] = filtered_df.index.astype(int)
    filtered_df.to_parquet(output_path)
    print(f"Original rows: {len(df)}, kept rows: {len(filtered_df)}")
    print(f"Saved parquet to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract low-reward samples for case study.")
    parser.add_argument("--data-path", required=True, help="Input dataset path (.json, .jsonl, or .parquet).")
    parser.add_argument("--reward-log", required=True, help="Reward tracking JSONL file.")
    parser.add_argument("--output", required=True, help="Destination file for the filtered dataset.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Upper bound: keep samples whose avg reward is below this value (default: 0.1).",
    )
    parser.add_argument(
        "--threshold-low",
        type=float,
        default=0.0,
        help="Lower bound: keep samples whose avg reward is >= this value (default: 0.0).",
    )
    args = parser.parse_args()

    if args.threshold_low >= args.threshold:
        raise ValueError("`threshold-low` must be smaller than `threshold`.")

    print(f"Parsing rewards from {args.reward_log}")
    keep_indices = collect_low_reward_indices(args.reward_log, args.threshold, args.threshold_low)

    input_path = Path(args.data_path)
    suffix = input_path.suffix.lower()
    print(f"Selecting hard samples from {args.data_path}")

    if suffix == ".json":
        keep_from_json(args.data_path, args.output, keep_indices)
    elif suffix == ".jsonl":
        keep_from_jsonl(args.data_path, args.output, keep_indices)
    elif suffix == ".parquet":
        keep_from_parquet(args.data_path, args.output, keep_indices)
    else:
        raise ValueError(f"Unsupported data format: {suffix}")

    print("Extraction complete.")


if __name__ == "__main__":
    main()
