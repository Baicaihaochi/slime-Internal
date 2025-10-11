#!/usr/bin/env python3
"""
Q-Tuning Data Pruning Analysis Script

This script implements the Q-Tuning pruning method from the paper:
"Winning the Pruning Gamble: A Unified Approach to Joint Sample and Token Pruning"

It processes math and code samples through two stages:
1. Sample-Level Pruning: Classify samples into Q1-Q4 quadrants based on PPL and Entropy
2. Token-Level Pruning: Prune high-PPL tokens from Q2 samples only

Output:
- stage1_kept.json: Samples retained after stage 1 (Q2 + Q4)
- stage1_removed.json: Samples removed in stage 1 (Q1 + Q3)
- stage2_final.json: Final samples after token pruning
- stage2_pruned_tokens.json: Visualization of removed tokens in Q2 samples
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add slime to path
SLIME_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SLIME_ROOT))


class QTuningAnalyzer:
    def __init__(
        self,
        model_path: str,
        data_path: str,
        output_dir: str,
        sample_keep_ratio: float = 0.5,
        token_keep_ratio: float = 0.7,
        neighbor_lambda: float = 0.5,
        ignore_special_tokens: bool = False,
        special_token_pairs: List[Tuple[str, str]] = None,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_keep_ratio = sample_keep_ratio
        self.token_keep_ratio = token_keep_ratio
        self.neighbor_lambda = neighbor_lambda

        # Long CoT special token handling
        self.ignore_special_tokens = ignore_special_tokens
        self.special_token_pairs = special_token_pairs or [
            ("<think>", "</think>"),
            ("<answer>", "</answer>"),
        ]

        print(f"Loading model from {model_path}...")

        # Debug: Show how special tokens are tokenized
        if self.ignore_special_tokens:
            print("\nSpecial token tokenization preview:")
            temp_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            for start_tok, end_tok in self.special_token_pairs:
                start_ids = temp_tokenizer.encode(start_tok, add_special_tokens=False)
                end_ids = temp_tokenizer.encode(end_tok, add_special_tokens=False)
                start_tokens = [temp_tokenizer.decode([tid]) for tid in start_ids]
                end_tokens = [temp_tokenizer.decode([tid]) for tid in end_ids]
                print(f"  {start_tok:20s} → {start_ids} = {start_tokens}")
                print(f"  {end_tok:20s} → {end_ids} = {end_tokens}")
            print()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Metal (MPS)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU (will be slow)")

        # Load model without device_map (simpler for single GPU/MPS)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device.type != "cpu" else torch.float32,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}!")

    def load_samples(self, n_math: int = 100, n_code: int = 100) -> Dict[str, List[Dict]]:
        """
        Load n_math math samples and n_code code samples from the dataset.

        Args:
            n_math: Number of math samples to load. Set to -1 for all math samples.
            n_code: Number of code samples to load. Set to -1 for all code samples.
        """
        print(f"\nLoading samples from {self.data_path}...")

        samples = {"math": [], "code": []}

        # -1 means load all samples
        load_all_math = (n_math == -1)
        load_all_code = (n_code == -1)

        if load_all_math and load_all_code:
            print("Loading ALL samples from dataset...")
        elif load_all_math:
            print(f"Loading ALL math samples and {n_code} code samples...")
        elif load_all_code:
            print(f"Loading {n_math} math samples and ALL code samples...")
        else:
            print(f"Loading {n_math} math samples and {n_code} code samples...")

        # Load the JSON data
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # The data structure is: {"problem": {"0": ..., "1": ...}, "category_": {"0": "math", ...}, "conversations": {"0": [...], ...}}
        # Convert to list of samples
        num_samples = len(data.get("problem", {}))
        print(f"Dataset contains {num_samples} samples")

        sample_list = []
        for idx in range(num_samples):
            idx_str = str(idx)

            # Safely get metadata - ensure it's a dict
            metadata = data.get("metadata", {})
            if metadata is None:
                metadata = {}
            sample_metadata = metadata.get(idx_str, {})
            if sample_metadata is None:
                sample_metadata = {}

            sample = {
                "id": idx,
                "problem": data.get("problem", {}).get(idx_str, ""),
                "category": data.get("category_", {}).get(idx_str, ""),
                "conversations": data.get("conversations", {}).get(idx_str, []),
                "metadata": sample_metadata,
            }

            sample_list.append(sample)

        print(f"Converted to {len(sample_list)} samples, filtering by category...")

        # Math categories: "math", "math-OT3", "Nemotron-math"
        # Code categories: "code-OT", "code-OT3", "Nemotron-code"
        math_keywords = ["math"]
        code_keywords = ["code"]

        # Filter samples by category
        for sample in tqdm(sample_list, desc="Filtering samples"):
            category = sample.get("category", "")

            # Check if it's a math sample
            is_math = any(keyword in category for keyword in math_keywords)
            # Check if it's a code sample
            is_code = any(keyword in category for keyword in code_keywords)

            if is_math and (load_all_math or len(samples["math"]) < n_math):
                samples["math"].append(sample)
            elif is_code and (load_all_code or len(samples["code"]) < n_code):
                samples["code"].append(sample)

            # Early exit if we have enough samples (only when not loading all)
            if not load_all_math and not load_all_code:
                if len(samples["math"]) >= n_math and len(samples["code"]) >= n_code:
                    break

        print(f"Collected {len(samples['math'])} math samples and {len(samples['code'])} code samples")
        return samples

    def _find_special_token_ranges(self, text: str) -> List[Tuple[int, int]]:
        """
        Find character ranges of special token pairs in text.
        Returns list of (start_idx, end_idx) tuples to ignore.
        """
        ignore_ranges = []
        for start_token, end_token in self.special_token_pairs:
            start_idx = 0
            while True:
                start_pos = text.find(start_token, start_idx)
                if start_pos == -1:
                    break
                end_pos = text.find(end_token, start_pos + len(start_token))
                if end_pos == -1:
                    # No matching end token, ignore from start to end of text
                    ignore_ranges.append((start_pos, len(text)))
                    break
                else:
                    # Found pair, ignore from start_token to end of end_token
                    ignore_ranges.append((start_pos, end_pos + len(end_token)))
                    start_idx = end_pos + len(end_token)

        # Merge overlapping ranges
        if ignore_ranges:
            ignore_ranges.sort()
            merged = [ignore_ranges[0]]
            for start, end in ignore_ranges[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            return merged
        return []

    def _tokenize_special_markers(self) -> Dict[str, List[int]]:
        """
        Pre-tokenize special marker strings to get their token IDs.
        Returns dict mapping marker string to token ID sequence.
        """
        marker_tokens = {}
        for start_marker, end_marker in self.special_token_pairs:
            # Tokenize without special tokens
            start_ids = self.tokenizer.encode(start_marker, add_special_tokens=False)
            end_ids = self.tokenizer.encode(end_marker, add_special_tokens=False)
            marker_tokens[start_marker] = start_ids
            marker_tokens[end_marker] = end_ids
        return marker_tokens

    def _find_special_token_id_ranges(
        self, token_ids: List[int], marker_tokens: Dict[str, List[int]]
    ) -> List[Tuple[int, int]]:
        """
        Find token index ranges that correspond to special markers.
        Returns list of (start_idx, end_idx) tuples to ignore.
        """
        ignore_ranges = []

        for start_marker, end_marker in self.special_token_pairs:
            start_pattern = marker_tokens[start_marker]
            end_pattern = marker_tokens[end_marker]

            # Find all occurrences of start pattern
            i = 0
            while i <= len(token_ids) - len(start_pattern):
                # Check if start pattern matches at position i
                if token_ids[i:i+len(start_pattern)] == start_pattern:
                    start_idx = i

                    # Look for matching end pattern
                    j = start_idx + len(start_pattern)
                    found_end = False

                    while j <= len(token_ids) - len(end_pattern):
                        if token_ids[j:j+len(end_pattern)] == end_pattern:
                            end_idx = j + len(end_pattern)  # Include end marker
                            ignore_ranges.append((start_idx, end_idx))
                            found_end = True
                            i = end_idx  # Skip past this range
                            break
                        j += 1

                    if not found_end:
                        # No matching end, ignore from start to end of sequence
                        ignore_ranges.append((start_idx, len(token_ids)))
                        break

                    continue
                i += 1

        # Merge overlapping ranges
        if ignore_ranges:
            ignore_ranges.sort()
            merged = [ignore_ranges[0]]
            for start, end in ignore_ranges[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            return merged

        return []

    def _create_token_mask(self, response_token_ids: List[int]) -> List[bool]:
        """
        Create a boolean mask for response tokens.
        True = include in PPL/entropy computation, False = ignore.

        Uses token-level matching instead of text matching to handle
        cases where special markers are split across multiple tokens.
        """
        if not self.ignore_special_tokens:
            return [True] * len(response_token_ids)

        # Get token patterns for special markers
        marker_tokens = self._tokenize_special_markers()

        # Find token ranges to ignore
        ignore_ranges = self._find_special_token_id_ranges(response_token_ids, marker_tokens)

        if not ignore_ranges:
            return [True] * len(response_token_ids)

        # Create mask based on token indices
        token_mask = [True] * len(response_token_ids)
        for start_idx, end_idx in ignore_ranges:
            for i in range(start_idx, min(end_idx, len(token_mask))):
                token_mask[i] = False

        return token_mask

    def compute_ppl_and_entropy(self, sample: Dict) -> Tuple[float, float, List[float], List[float], List[bool]]:
        """
        Compute perplexity and entropy for a sample.

        Returns:
            (sample_ppl, sample_entropy, token_ppls, token_entropies, token_inclusion_mask)
            token_inclusion_mask: True for tokens to include in pruning consideration
        """
        # Extract prompt and response from conversations
        prompt = ""
        response = ""

        if "conversations" in sample and sample["conversations"]:
            conversations = sample["conversations"]
            for msg in conversations:
                if msg.get("from") == "human":
                    prompt += msg.get("value", "")
                elif msg.get("from") == "gpt":
                    response += msg.get("value", "")

        if not prompt or not response:
            # Return high values to mark as Q1 (noise)
            return 1000.0, 10.0, [], [], []

        # Tokenize
        full_text = prompt + response
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True, return_tensors="pt")

        # Move to device
        full_ids = full_ids.to(self.device)
        prompt_length = prompt_ids.shape[1]

        # Get response token IDs
        response_token_ids = full_ids[0, prompt_length:].tolist()

        # Create mask for special tokens (token-level matching)
        token_inclusion_mask = self._create_token_mask(response_token_ids)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(full_ids, labels=full_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # Compute token-level metrics (only for response tokens)
        token_ppls = []
        token_entropies = []
        token_nlls = []

        for i in range(prompt_length, full_ids.shape[1]):
            token_idx = i - prompt_length

            # Get token logits and compute log probs
            token_logits = logits[0, i-1, :]  # Predict token at position i
            log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
            probs = torch.exp(log_probs)

            # True token
            true_token_id = full_ids[0, i].item()
            token_nll = -log_probs[true_token_id].item()

            # Token perplexity
            token_ppl = np.exp(token_nll)
            token_ppls.append(token_ppl)

            # Token entropy: -sum(p * log(p))
            entropy = -(probs * log_probs).sum().item()
            token_entropies.append(entropy)

            # Only include in sample-level metrics if not in special token range
            if token_idx < len(token_inclusion_mask) and token_inclusion_mask[token_idx]:
                token_nlls.append(token_nll)

        # Sample-level metrics (average over non-special tokens only)
        if len(token_nlls) > 0:
            sample_ppl = np.exp(np.mean(token_nlls))
            # Filter entropies too
            filtered_entropies = [
                ent for i, ent in enumerate(token_entropies)
                if i < len(token_inclusion_mask) and token_inclusion_mask[i]
            ]
            sample_entropy = np.mean(filtered_entropies) if filtered_entropies else np.mean(token_entropies)
        else:
            sample_ppl = 1000.0
            sample_entropy = 10.0

        return sample_ppl, sample_entropy, token_ppls, token_entropies, token_inclusion_mask

    def classify_quadrant(
        self, ppl: float, entropy: float,
        ppl_low: float, ppl_high: float,
        ent_low: float, ent_high: float
    ) -> str:
        """
        Classify sample into Q1-Q4 based on thresholds.

        Uses strict conditions to ensure proper quadrant assignment:
        - Q1 (Harmful Noise): High PPL + High Entropy
        - Q2 (Valuable Misconception): High PPL + Low Entropy
        - Q3 (Redundant Knowledge): Low PPL + Low Entropy
        - Q4 (Calibration Data): Low PPL + High Entropy
        """
        # Determine PPL category
        if ppl >= ppl_high:
            ppl_category = "high"
        elif ppl < ppl_low:
            ppl_category = "low"
        else:
            ppl_category = "mid"

        # Determine Entropy category
        if entropy >= ent_high:
            ent_category = "high"
        elif entropy < ent_low:
            ent_category = "low"
        else:
            ent_category = "mid"

        # Classify based on combination
        if ppl_category == "high" and ent_category == "high":
            return "Q1"  # Harmful Noise
        elif ppl_category == "high" and ent_category == "low":
            return "Q2"  # Valuable Misconception
        elif ppl_category == "low" and ent_category == "low":
            return "Q3"  # Redundant Knowledge
        elif ppl_category == "low" and ent_category == "high":
            return "Q4"  # Calibration Data
        else:
            # Mid-range samples: assign to nearest quadrant based on which boundary they're closer to
            # This handles edge cases where samples fall in the middle region
            if ppl_category == "high" and ent_category == "mid":
                # High PPL, mid entropy - lean towards Q2 (misconception)
                return "Q2"
            elif ppl_category == "low" and ent_category == "mid":
                # Low PPL, mid entropy - lean towards Q3 (redundant)
                return "Q3"
            elif ppl_category == "mid" and ent_category == "high":
                # Mid PPL, high entropy - lean towards Q4 (calibration)
                return "Q4"
            elif ppl_category == "mid" and ent_category == "low":
                # Mid PPL, low entropy - lean towards Q3 (redundant)
                return "Q3"
            else:
                # Mid PPL, mid entropy - default to Q4 (calibration, conservative)
                return "Q4"

    def bisect_search_thresholds(
        self, ppls: List[float], entropies: List[float]
    ) -> Tuple[float, float, float, float]:
        """
        Bisection search to find thresholds that keep sample_keep_ratio samples in Q2+Q4.

        Returns:
            (ppl_low, ppl_high, ent_low, ent_high)
        """
        ppls = np.array(ppls)
        entropies = np.array(entropies)

        # Dynamic upper bound based on target keep ratio
        # Maximum alpha/beta that still allows keeping target ratio
        # When alpha=beta=0.5, all samples become "mid" range
        max_quantile = min(0.495, (1.0 - self.sample_keep_ratio) / 2.0 + 0.02)

        alpha_low, alpha_high = 0.0, max_quantile
        beta_low, beta_high = 0.0, max_quantile

        n_iterations = 15  # Increased for better convergence
        best_ratio = 0.0
        best_thresholds = None

        for _ in range(n_iterations):
            alpha = (alpha_low + alpha_high) / 2
            beta = (beta_low + beta_high) / 2

            # Compute thresholds
            ppl_low = np.quantile(ppls, alpha)
            ppl_high = np.quantile(ppls, 1 - alpha)
            ent_low = np.quantile(entropies, beta)
            ent_high = np.quantile(entropies, 1 - beta)

            # Count samples in Q2 and Q4
            q2_q4_count = 0
            for ppl, ent in zip(ppls, entropies):
                quad = self.classify_quadrant(ppl, ent, ppl_low, ppl_high, ent_low, ent_high)
                if quad in ["Q2", "Q4"]:
                    q2_q4_count += 1

            ratio = q2_q4_count / len(ppls)

            # Track best result
            if abs(ratio - self.sample_keep_ratio) < abs(best_ratio - self.sample_keep_ratio):
                best_ratio = ratio
                best_thresholds = (ppl_low, ppl_high, ent_low, ent_high)

            # Binary search adjustment
            if ratio < self.sample_keep_ratio:
                # Too few kept, relax thresholds (decrease alpha/beta)
                alpha_high = alpha
                beta_high = beta
            else:
                # Too many kept, tighten thresholds (increase alpha/beta)
                alpha_low = alpha
                beta_low = beta

            # Early stopping if close enough
            if abs(ratio - self.sample_keep_ratio) < 0.02:  # Within 2%
                break

        # Use best found thresholds if final iteration isn't optimal
        if best_thresholds and abs(best_ratio - self.sample_keep_ratio) < abs(ratio - self.sample_keep_ratio):
            return best_thresholds

        return ppl_low, ppl_high, ent_low, ent_high

    def neighbor_aware_token_scoring(
        self, token_ppls: List[float]
    ) -> List[float]:
        """Compute neighbor-aware token scores."""
        scores = []
        for i in range(len(token_ppls)):
            ppl_i = token_ppls[i]

            # Get neighbor PPLs
            ppl_prev = token_ppls[i-1] if i > 0 else ppl_i
            ppl_next = token_ppls[i+1] if i < len(token_ppls) - 1 else ppl_i

            # Compute score
            score = (1 - self.neighbor_lambda) * ppl_i + \
                    self.neighbor_lambda * (ppl_prev + ppl_next) / 2
            scores.append(score)

        return scores

    def stage1_sample_pruning(
        self, samples: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Stage 1: Sample-level pruning based on EU Plane.

        Returns:
            {
                "kept": [...],  # Q2 + Q4 samples
                "removed": {...},  # Q1 and Q3 samples by quadrant
                "quadrants": {...},  # All quadrants for comparison
                "statistics": {...}
            }
        """
        print("\n" + "="*80)
        print("STAGE 1: SAMPLE-LEVEL PRUNING")
        print("="*80)

        all_samples = samples["math"] + samples["code"]

        # Compute PPL and Entropy for all samples
        print("\nComputing perplexity and entropy...")
        ppls = []
        entropies = []
        enriched_samples = []

        for sample in tqdm(all_samples, desc="Computing metrics"):
            ppl, entropy, token_ppls, token_entropies, token_mask = self.compute_ppl_and_entropy(sample)

            # Add metrics to sample metadata
            if "metadata" not in sample or sample["metadata"] is None:
                sample["metadata"] = {}
            sample["metadata"]["ppl"] = float(ppl)
            sample["metadata"]["entropy"] = float(entropy)
            sample["metadata"]["token_ppls"] = [float(p) for p in token_ppls]
            sample["metadata"]["token_entropies"] = [float(e) for e in token_entropies]
            sample["metadata"]["special_token_mask"] = token_mask  # Save for stage2

            ppls.append(ppl)
            entropies.append(entropy)
            enriched_samples.append(sample)

        # Bisection search for thresholds
        print(f"\nSearching for thresholds (target keep ratio: {self.sample_keep_ratio})...")
        ppl_low, ppl_high, ent_low, ent_high = self.bisect_search_thresholds(ppls, entropies)

        print(f"Thresholds found:")
        print(f"  PPL:     [{ppl_low:.3f}, {ppl_high:.3f}]")
        print(f"  Entropy: [{ent_low:.3f}, {ent_high:.3f}]")

        # Classify samples
        print("\nClassifying samples into quadrants...")
        quadrants = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}

        for sample, ppl, entropy in zip(enriched_samples, ppls, entropies):
            quad = self.classify_quadrant(ppl, entropy, ppl_low, ppl_high, ent_low, ent_high)
            sample["metadata"]["quadrant"] = quad
            quadrants[quad].append(sample)

        # Statistics
        stats = {
            "total_samples": len(enriched_samples),
            "Q1_count": len(quadrants["Q1"]),
            "Q2_count": len(quadrants["Q2"]),
            "Q3_count": len(quadrants["Q3"]),
            "Q4_count": len(quadrants["Q4"]),
            "kept_count": len(quadrants["Q2"]) + len(quadrants["Q4"]),
            "removed_count": len(quadrants["Q1"]) + len(quadrants["Q3"]),
            "actual_keep_ratio": (len(quadrants["Q2"]) + len(quadrants["Q4"])) / len(enriched_samples),
            "thresholds": {
                "ppl_low": float(ppl_low),
                "ppl_high": float(ppl_high),
                "ent_low": float(ent_low),
                "ent_high": float(ent_high),
            }
        }

        print(f"\nStage 1 Results:")
        print(f"  Q1 (Harmful Noise):         {stats['Q1_count']:3d} samples - REMOVED")
        print(f"  Q2 (Valuable Misconception): {stats['Q2_count']:3d} samples - KEPT (will prune tokens)")
        print(f"  Q3 (Redundant Knowledge):    {stats['Q3_count']:3d} samples - REMOVED")
        print(f"  Q4 (Calibration Data):       {stats['Q4_count']:3d} samples - KEPT (full)")
        print(f"  Total kept: {stats['kept_count']}/{stats['total_samples']} ({stats['actual_keep_ratio']:.1%})")

        return {
            "kept": quadrants["Q2"] + quadrants["Q4"],
            "removed": {"Q1": quadrants["Q1"], "Q3": quadrants["Q3"]},
            "quadrants": quadrants,
            "statistics": stats,
        }

    def stage2_token_pruning(
        self, stage1_kept: List[Dict]
    ) -> Dict[str, Any]:
        """
        Stage 2: Token-level pruning for Q2 samples only.

        Returns:
            {
                "final_samples": [...],
                "pruned_visualizations": [...],
                "statistics": {...}
            }
        """
        print("\n" + "="*80)
        print("STAGE 2: TOKEN-LEVEL PRUNING (Q2 only)")
        print("="*80)

        final_samples = []
        pruned_visualizations = []

        q2_count = sum(1 for s in stage1_kept if s["metadata"]["quadrant"] == "Q2")
        q4_count = sum(1 for s in stage1_kept if s["metadata"]["quadrant"] == "Q4")

        print(f"\nProcessing {q2_count} Q2 samples (will prune) and {q4_count} Q4 samples (keep full)...")

        total_tokens_before = 0
        total_tokens_after = 0

        for sample in tqdm(stage1_kept, desc="Token pruning"):
            quadrant = sample["metadata"]["quadrant"]

            if quadrant == "Q4":
                # Keep all tokens
                sample["metadata"]["tokens_kept"] = "all"
                final_samples.append(sample)

            elif quadrant == "Q2":
                # Apply token pruning
                token_ppls = sample["metadata"]["token_ppls"]
                special_token_mask = sample["metadata"].get("special_token_mask", None)

                if len(token_ppls) == 0:
                    final_samples.append(sample)
                    continue

                total_tokens_before += len(token_ppls)

                # Compute neighbor-aware scores
                scores = self.neighbor_aware_token_scoring(token_ppls)

<<<<<<< HEAD
                # Determine threshold (keep top token_keep_ratio tokens)
                n_keep = max(1, int(len(scores) * self.token_keep_ratio))
<<<<<<< HEAD
                score_threshold = sorted(scores)[n_keep - 1]

                # Create token mask
                token_mask = [1 if s <= score_threshold else 0 for s in scores]
=======
                score_threshold = sorted(scores, reverse=True)[n_keep - 1]

                # Create token mask
                token_mask = [1 if s >= score_threshold else 0 for s in scores]
>>>>>>> f75c7c5 (Slime length penalty (#1))
=======
                # If special token handling is enabled, force keep special tokens
                if self.ignore_special_tokens and special_token_mask:
                    # Count how many prunable tokens we have (excluding special tokens)
                    prunable_indices = [i for i in range(len(scores))
                                       if i >= len(special_token_mask) or special_token_mask[i]]

                    if prunable_indices:
                        # Determine how many prunable tokens to keep
                        n_keep_prunable = max(1, int(len(prunable_indices) * self.token_keep_ratio))

                        # Get scores only for prunable tokens
                        prunable_scores = [(i, scores[i]) for i in prunable_indices]
                        prunable_scores.sort(key=lambda x: x[1])  # Sort by score

                        # Select indices to keep (lowest scores)
                        keep_indices = set(idx for idx, _ in prunable_scores[:n_keep_prunable])

                        # Create token mask: keep special tokens + selected prunable tokens
                        token_mask = []
                        for i in range(len(scores)):
                            if i < len(special_token_mask) and not special_token_mask[i]:
                                # This is a special token, always keep
                                token_mask.append(1)
                            elif i in keep_indices or i >= len(special_token_mask):
                                # Selected for keeping or beyond mask range
                                token_mask.append(1 if i in keep_indices else 0)
                            else:
                                token_mask.append(0)
                    else:
                        # All tokens are special tokens, keep all
                        token_mask = [1] * len(scores)
                else:
                    # No special token handling, use normal pruning
                    n_keep = max(1, int(len(scores) * self.token_keep_ratio))
                    score_threshold = sorted(scores)[n_keep - 1]
                    token_mask = [1 if s <= score_threshold else 0 for s in scores]

>>>>>>> d6bfd86 (qtuning test bug fix Q2Q4 sample & Long CoT token pruning)
                sample["metadata"]["token_mask"] = token_mask
                sample["metadata"]["tokens_kept"] = sum(token_mask)
                sample["metadata"]["tokens_removed"] = len(token_mask) - sum(token_mask)

                total_tokens_after += sum(token_mask)

                # Create visualization
                vis = self.create_token_visualization(sample)
                pruned_visualizations.append(vis)

                final_samples.append(sample)

        stats = {
            "q2_samples": q2_count,
            "q4_samples": q4_count,
            "total_tokens_before": total_tokens_before,
            "total_tokens_after": total_tokens_after,
            "tokens_removed": total_tokens_before - total_tokens_after,
            "token_compression_ratio": total_tokens_after / total_tokens_before if total_tokens_before > 0 else 1.0,
        }

        print(f"\nStage 2 Results:")
        print(f"  Q2 samples processed: {q2_count}")
        print(f"  Q4 samples kept full: {q4_count}")
        print(f"  Tokens before pruning: {stats['total_tokens_before']}")
        print(f"  Tokens after pruning:  {stats['total_tokens_after']}")
        print(f"  Token compression:     {stats['token_compression_ratio']:.1%}")

        return {
            "final_samples": final_samples,
            "pruned_visualizations": pruned_visualizations,
            "statistics": stats,
        }

    def create_token_visualization(self, sample: Dict) -> Dict:
        """Create a visualization showing removed tokens."""
        # Extract response from conversations
        response = ""
        if "conversations" in sample and sample["conversations"]:
            for msg in sample["conversations"]:
                if msg.get("from") == "gpt":
                    response += msg.get("value", "")

        # Tokenize response
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        response_text_tokens = [self.tokenizer.decode([t]) for t in response_tokens]

        token_mask = sample["metadata"].get("token_mask", [])
        token_ppls = sample["metadata"].get("token_ppls", [])

        # Align (may have length mismatch, take minimum)
        min_len = min(len(response_text_tokens), len(token_mask), len(token_ppls))

        visualization = {
            "sample_id": sample.get("id", "unknown"),
            "quadrant": sample["metadata"]["quadrant"],
            "tokens": []
        }

        for i in range(min_len):
            visualization["tokens"].append({
                "text": response_text_tokens[i],
                "kept": bool(token_mask[i]),
                "ppl": float(token_ppls[i]),
            })

        return visualization

    def generate_html_visualization(
        self, stage1_result: Dict, stage2_result: Dict
    ) -> str:
        """Generate comprehensive HTML visualization comparing both stages."""
        quadrants = stage1_result["quadrants"]
        stats1 = stage1_result["statistics"]
        stats2 = stage2_result["statistics"]

        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Q-Tuning Pruning Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .section {
            background-color: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .section-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #2c3e50;
            border-left: 5px solid #667eea;
            padding-left: 15px;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        .quadrant-box {
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .q1-box { border-color: #e74c3c; background-color: #fdecea; }
        .q2-box { border-color: #f39c12; background-color: #fef5e7; }
        .q3-box { border-color: #95a5a6; background-color: #ecf0f1; }
        .q4-box { border-color: #3498db; background-color: #ebf5fb; }
        .quadrant-header {
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .quadrant-count {
            background-color: rgba(0,0,0,0.1);
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 14px;
        }
        .quadrant-desc {
            font-style: italic;
            color: #666;
            margin-bottom: 15px;
        }
        .sample-preview {
            background-color: white;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
            font-size: 13px;
            max-height: 150px;
            overflow-y: auto;
        }
        .sample-content {
            color: #444;
            line-height: 1.6;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 10px;
            font-size: 13px;
        }
        .metric {
            background-color: rgba(255,255,255,0.5);
            padding: 8px;
            border-radius: 5px;
        }
        .metric-label {
            font-weight: bold;
            color: #555;
        }
        .token {
            display: inline-block;
            padding: 3px 6px;
            margin: 2px;
            border-radius: 3px;
            font-family: 'Consolas', monospace;
            font-size: 13px;
        }
        .token-kept {
            background-color: #2ecc71;
            color: white;
        }
        .token-removed {
            background-color: #e74c3c;
            color: white;
            text-decoration: line-through;
            opacity: 0.7;
        }
        .stage2-sample {
            background-color: white;
            border: 2px solid #f39c12;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .stage2-header {
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
            font-size: 16px;
        }
        .token-stats {
            background-color: #fef5e7;
            padding: 12px;
            border-radius: 5px;
            margin-top: 15px;
            font-size: 14px;
        }
        .legend {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .legend-title {
            font-weight: bold;
            margin-right: 10px;
        }
        .stats-summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 15px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 14px;
            opacity: 0.9;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Q-Tuning Pruning Analysis</h1>
        <p>Comprehensive visualization of two-stage data pruning: Sample-level (Stage 1) and Token-level (Stage 2)</p>
    </div>

    <div class="stats-summary">
        <h2 style="margin-top: 0;">Overall Statistics</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{stats1['total_samples']}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats1['kept_count']}</div>
                <div class="stat-label">Kept After Stage 1</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats1['actual_keep_ratio']:.1%}</div>
                <div class="stat-label">Sample Keep Ratio</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats2['token_compression_ratio']:.1%}</div>
                <div class="stat-label">Token Compression</div>
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Stage 1: Sample-Level Pruning (EU Plane Quadrants)</div>
        <p style="color: #666; margin-bottom: 20px;">
            Samples are classified based on Perplexity (error) and Entropy (uncertainty).
            <strong>Q2 and Q4 are kept</strong>, while <strong>Q1 and Q3 are removed</strong>.
        </p>

        <div class="comparison-grid">
"""

        # Generate quadrant boxes with sample previews
        quadrant_info = {
            "Q1": ("Harmful Noise", "High PPL + High Entropy", "REMOVED", "q1-box"),
            "Q2": ("Valuable Misconception", "High PPL + Low Entropy", "KEPT → Token Pruning", "q2-box"),
            "Q3": ("Redundant Knowledge", "Low PPL + Low Entropy", "REMOVED", "q3-box"),
            "Q4": ("Calibration Data", "Low PPL + High Entropy", "KEPT (Full)", "q4-box"),
        }

        for quad_name in ["Q1", "Q2", "Q3", "Q4"]:
            title, desc, action, css_class = quadrant_info[quad_name]
            samples = quadrants[quad_name]
            count = len(samples)

            html += f"""
            <div class="quadrant-box {css_class}">
                <div class="quadrant-header">
                    <span>{quad_name}: {title}</span>
                    <span class="quadrant-count">{count} samples</span>
                </div>
                <div class="quadrant-desc">{desc} → {action}</div>
"""

            # Show first sample as preview
            if samples:
                sample = samples[0]
                ppl = sample["metadata"].get("ppl", 0)
                entropy = sample["metadata"].get("entropy", 0)

                # Extract text preview
                text_preview = ""
                if "conversations" in sample and sample["conversations"]:
                    for msg in sample["conversations"][:2]:
                        role = "User" if msg.get("from") == "human" else "Assistant"
                        content = msg.get("value", "")[:200]
                        text_preview += f"<strong>{role}:</strong> {content}...<br>"

                html += f"""
                <div class="sample-preview">
                    <div class="sample-content">{text_preview}</div>
                    <div class="metrics">
                        <div class="metric">
                            <span class="metric-label">PPL:</span> {ppl:.2f}
                        </div>
                        <div class="metric">
                            <span class="metric-label">Entropy:</span> {entropy:.2f}
                        </div>
                    </div>
                </div>
"""

            html += """
            </div>
"""

        html += """
        </div>
    </div>

    <div class="section">
        <div class="section-title">Stage 2: Token-Level Pruning (Q2 Samples Only)</div>
        <p style="color: #666; margin-bottom: 20px;">
            For Q2 samples (Valuable Misconceptions), we apply neighbor-aware token pruning to remove high-perplexity tokens while keeping low-perplexity ones.
        </p>

        <div class="legend">
            <span class="legend-title">Legend:</span>
            <span class="token token-kept">Kept Token</span>
            <span class="token token-removed">Removed Token</span>
        </div>
"""

        # Show token pruning examples
        for i, vis in enumerate(stage2_result["pruned_visualizations"][:20]):
            html += f"""
        <div class="stage2-sample">
            <div class="stage2-header">Sample {i+1} (ID: {vis['sample_id']})</div>
            <div>
"""
            for token_info in vis["tokens"]:
                token_class = "token-kept" if token_info["kept"] else "token-removed"
                token_text = token_info["text"].replace(" ", "·").replace("<", "&lt;").replace(">", "&gt;")
                ppl = token_info["ppl"]
                html += f'<span class="token {token_class}" title="PPL: {ppl:.2f}">{token_text}</span>'

            kept = sum(1 for t in vis["tokens"] if t["kept"])
            removed = sum(1 for t in vis["tokens"] if not t["kept"])
            total = len(vis["tokens"])
            compression = kept / total * 100 if total > 0 else 0

            html += f"""
            </div>
            <div class="token-stats">
                <strong>Tokens:</strong> {kept} kept / {removed} removed / {total} total
                <strong style="margin-left: 20px;">Compression:</strong> {compression:.1f}%
            </div>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html

    def save_results(
        self,
        stage1_result: Dict,
        stage2_result: Dict
    ):
        """Save all results to output directory."""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)

        # Stage 1: kept samples
        stage1_kept_path = self.output_dir / "stage1_kept.json"
        with open(stage1_kept_path, 'w', encoding='utf-8') as f:
            json.dump(stage1_result["kept"], f, ensure_ascii=False, indent=2)
        print(f"Saved {len(stage1_result['kept'])} kept samples to {stage1_kept_path}")

        # Stage 1: removed samples
        stage1_removed_path = self.output_dir / "stage1_removed.json"
        with open(stage1_removed_path, 'w', encoding='utf-8') as f:
            json.dump(stage1_result["removed"], f, ensure_ascii=False, indent=2)
        removed_count = len(stage1_result["removed"]["Q1"]) + len(stage1_result["removed"]["Q3"])
        print(f"Saved {removed_count} removed samples to {stage1_removed_path}")

        # Stage 2: final samples
        stage2_final_path = self.output_dir / "stage2_final.json"
        with open(stage2_final_path, 'w', encoding='utf-8') as f:
            json.dump(stage2_result["final_samples"], f, ensure_ascii=False, indent=2)
        print(f"Saved {len(stage2_result['final_samples'])} final samples to {stage2_final_path}")

        # Stage 2: token pruning visualizations
        stage2_vis_path = self.output_dir / "stage2_pruned_tokens_visualization.json"
        with open(stage2_vis_path, 'w', encoding='utf-8') as f:
            json.dump(stage2_result["pruned_visualizations"], f, ensure_ascii=False, indent=2)
        print(f"Saved {len(stage2_result['pruned_visualizations'])} token visualizations to {stage2_vis_path}")

        # HTML visualization
        html_path = self.output_dir / "token_pruning_visualization.html"
        html_content = self.generate_html_visualization(stage1_result, stage2_result)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Saved HTML visualization to {html_path}")

        # Statistics summary
        summary = {
            "stage1": stage1_result["statistics"],
            "stage2": stage2_result["statistics"],
        }
        summary_path = self.output_dir / "summary_statistics.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved statistics summary to {summary_path}")

        print("\n" + "="*80)
        print("ALL RESULTS SAVED SUCCESSFULLY!")
        print(f"\n📊 View visualization: file://{html_path.absolute()}")
        print("="*80)

    def run(self, n_math: int = 100, n_code: int = 100):
        """
        Run the full Q-Tuning analysis pipeline.

        Args:
            n_math: Number of math samples. Set to -1 for all math samples.
            n_code: Number of code samples. Set to -1 for all code samples.
        """
        # Load samples
        samples = self.load_samples(n_math=n_math, n_code=n_code)

        # Stage 1: Sample-level pruning
        stage1_result = self.stage1_sample_pruning(samples)

        # Stage 2: Token-level pruning
        stage2_result = self.stage2_token_pruning(stage1_result["kept"])

        # Save results
        self.save_results(stage1_result, stage2_result)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Q-Tuning Data Pruning Analysis")
    parser.add_argument("--model-path", type=str,
                       default="/Users/shuocai/Documents/code/iter_0010999__e8m0",
                       help="Path to the model")
    parser.add_argument("--data-path", type=str,
                       default="/Users/shuocai/Documents/code/cs_data/0726--57kmath_57kcode_34kscience_deduped--0.8-easy-math-code-final.json",
                       help="Path to the dataset")
    parser.add_argument("--output-dir", type=str,
                       default="/Users/shuocai/Downloads/slime/tests/q_tuning_analysis_output",
                       help="Output directory")
    parser.add_argument("--n-math", type=int, default=100,
                       help="Number of math samples to process. -1 for all samples.")
    parser.add_argument("--n-code", type=int, default=100,
                       help="Number of code samples to process. -1 for all samples.")
    parser.add_argument("--sample-keep-ratio", type=float, default=0.5,
                       help="Sample keep ratio (default: 0.5)")
    parser.add_argument("--token-keep-ratio", type=float, default=0.7,
                       help="Token keep ratio for Q2 samples (default: 0.7)")
    parser.add_argument("--neighbor-lambda", type=float, default=0.5,
                       help="Neighbor weight in token scoring (default: 0.5)")
    parser.add_argument("--ignore-special-tokens", action="store_true",
                       help="Ignore tokens within special token pairs (e.g., <think>...</think>) when computing PPL/Entropy")
    parser.add_argument("--special-token-pairs", type=str, nargs="+",
                       default=["<think>,</think>", "<answer>,</answer>"],
                       help="Special token pairs to ignore, format: 'start,end' (default: '<think>,</think>' '<answer>,</answer>')")

    args = parser.parse_args()

    # Parse special token pairs
    special_pairs = []
    for pair in args.special_token_pairs:
        parts = pair.split(",")
        if len(parts) == 2:
            special_pairs.append((parts[0], parts[1]))
        else:
            print(f"Warning: Invalid special token pair format: {pair}, skipping...")

    print(f"\n{'='*80}")
    print("Q-TUNING PRUNING ANALYSIS")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Sample keep ratio: {args.sample_keep_ratio}")
    print(f"Token keep ratio: {args.token_keep_ratio}")
    if args.ignore_special_tokens:
        print(f"Special token handling: ENABLED")
        print(f"  Ignoring tokens within: {special_pairs}")
    else:
        print(f"Special token handling: DISABLED")
    print(f"{'='*80}\n")

    # Create analyzer
    analyzer = QTuningAnalyzer(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_keep_ratio=args.sample_keep_ratio,
        token_keep_ratio=args.token_keep_ratio,
        neighbor_lambda=args.neighbor_lambda,
        ignore_special_tokens=args.ignore_special_tokens,
        special_token_pairs=special_pairs if special_pairs else None,
    )

    # Run analysis
    analyzer.run(n_math=args.n_math, n_code=args.n_code)


if __name__ == "__main__":
    main()
