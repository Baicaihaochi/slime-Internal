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
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_keep_ratio = sample_keep_ratio
        self.token_keep_ratio = token_keep_ratio
        self.neighbor_lambda = neighbor_lambda

        print(f"Loading model from {model_path}...")
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

    def compute_ppl_and_entropy(self, sample: Dict) -> Tuple[float, float, List[float], List[float]]:
        """
        Compute perplexity and entropy for a sample.

        Returns:
            (sample_ppl, sample_entropy, token_ppls, token_entropies)
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
            return 1000.0, 10.0, [], []

        # Tokenize
        full_text = prompt + response
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True, return_tensors="pt")

        # Move to device
        full_ids = full_ids.to(self.device)
        prompt_length = prompt_ids.shape[1]

        # Forward pass
        with torch.no_grad():
            outputs = self.model(full_ids, labels=full_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # Compute token-level metrics (only for response tokens)
        token_ppls = []
        token_entropies = []
        token_nlls = []

        for i in range(prompt_length, full_ids.shape[1]):
            # Get token logits and compute log probs
            token_logits = logits[0, i-1, :]  # Predict token at position i
            log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
            probs = torch.exp(log_probs)

            # True token
            true_token_id = full_ids[0, i].item()
            token_nll = -log_probs[true_token_id].item()
            token_nlls.append(token_nll)

            # Token perplexity
            token_ppl = np.exp(token_nll)
            token_ppls.append(token_ppl)

            # Token entropy: -sum(p * log(p))
            entropy = -(probs * log_probs).sum().item()
            token_entropies.append(entropy)

        # Sample-level metrics (average over response tokens)
        if len(token_nlls) > 0:
            sample_ppl = np.exp(np.mean(token_nlls))
            sample_entropy = np.mean(token_entropies)
        else:
            sample_ppl = 1000.0
            sample_entropy = 10.0

        return sample_ppl, sample_entropy, token_ppls, token_entropies

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

        alpha_low, alpha_high = 0.0, 0.49
        beta_low, beta_high = 0.0, 0.49

        n_iterations = 10
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

            if ratio < self.sample_keep_ratio:
                # Too few kept, relax thresholds
                alpha_low = alpha
                beta_low = beta
            else:
                # Too many kept, tighten thresholds
                alpha_high = alpha
                beta_high = beta

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
            ppl, entropy, token_ppls, token_entropies = self.compute_ppl_and_entropy(sample)

            # Add metrics to sample metadata
            if "metadata" not in sample or sample["metadata"] is None:
                sample["metadata"] = {}
            sample["metadata"]["ppl"] = float(ppl)
            sample["metadata"]["entropy"] = float(entropy)
            sample["metadata"]["token_ppls"] = [float(p) for p in token_ppls]
            sample["metadata"]["token_entropies"] = [float(e) for e in token_entropies]

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

                if len(token_ppls) == 0:
                    final_samples.append(sample)
                    continue

                total_tokens_before += len(token_ppls)

                # Compute neighbor-aware scores
                scores = self.neighbor_aware_token_scoring(token_ppls)

                # Determine threshold (keep top token_keep_ratio tokens)
                n_keep = max(1, int(len(scores) * self.token_keep_ratio))
                score_threshold = sorted(scores, reverse=True)[n_keep - 1]

                # Create token mask
                token_mask = [1 if s >= score_threshold else 0 for s in scores]
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

    def generate_html_visualization(self, stage2_result: Dict) -> str:
        """Generate an HTML file to visualize token pruning."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Q-Tuning Token Pruning Visualization</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .sample {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample-header {
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
        .token {
            display: inline-block;
            padding: 2px 5px;
            margin: 2px;
            border-radius: 3px;
            font-family: monospace;
        }
        .token-kept {
            background-color: #2ecc71;
            color: white;
        }
        .token-removed {
            background-color: #e74c3c;
            color: white;
            text-decoration: line-through;
        }
        .legend {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
        }
        .stats {
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 3px;
            margin-top: 10px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Q-Tuning Token Pruning Visualization</h1>
        <p>This page shows token-level pruning results for Q2 (Valuable Misconception) samples.</p>
    </div>

    <div class="legend">
        <div class="legend-item">
            <span class="token token-kept">Kept Token</span>
        </div>
        <div class="legend-item">
            <span class="token token-removed">Removed Token</span>
        </div>
    </div>
"""

        for i, vis in enumerate(stage2_result["pruned_visualizations"][:50]):  # Show first 50
            html += f"""
    <div class="sample">
        <div class="sample-header">Sample {i+1} (ID: {vis['sample_id']}, Quadrant: {vis['quadrant']})</div>
        <div>
"""
            for token_info in vis["tokens"]:
                token_class = "token-kept" if token_info["kept"] else "token-removed"
                token_text = token_info["text"].replace(" ", "·")  # Make spaces visible
                ppl = token_info["ppl"]
                html += f'<span class="token {token_class}" title="PPL: {ppl:.2f}">{token_text}</span>'

            kept_count = sum(1 for t in vis["tokens"] if t["kept"])
            removed_count = sum(1 for t in vis["tokens"] if not t["kept"])
            html += f"""
        </div>
        <div class="stats">
            Tokens: {kept_count} kept / {removed_count} removed / {len(vis["tokens"])} total
            (compression: {kept_count/len(vis["tokens"])*100:.1f}%)
        </div>
    </div>
"""

        html += """
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
        html_content = self.generate_html_visualization(stage2_result)
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

    args = parser.parse_args()

    # Create analyzer
    analyzer = QTuningAnalyzer(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_keep_ratio=args.sample_keep_ratio,
        token_keep_ratio=args.token_keep_ratio,
        neighbor_lambda=args.neighbor_lambda,
    )

    # Run analysis
    analyzer.run(n_math=args.n_math, n_code=args.n_code)


if __name__ == "__main__":
    main()
