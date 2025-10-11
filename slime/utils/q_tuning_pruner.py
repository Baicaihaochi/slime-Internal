"""
Q-Tuning: Dynamic Data Pruning for Efficient LLM Fine-Tuning

This module implements the Q-Tuning algorithm from "Winning the Pruning Gamble" (arXiv:2509.23873).
Q-Tuning performs joint sample and token pruning based on the Error-Uncertainty (EU) Plane, which
categorizes training data into four quadrants using perplexity (error) and entropy (uncertainty).

Reference: https://arxiv.org/abs/2509.23873
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class QTuningPruner:
    """
    Q-Tuning dynamic data pruner implementing the EU Plane framework.

    The pruner operates in two stages:
    1. Sample-level pruning: Classify samples into Q1-Q4 based on PPL and Entropy
    2. Token-level pruning: Apply neighbor-aware token pruning to Q2 samples

    Quadrants:
    - Q1 (Harmful Noise): High PPL + High Entropy → Remove
    - Q2 (Valuable Misconception): High PPL + Low Entropy → Keep + Token Pruning
    - Q3 (Redundant Knowledge): Low PPL + Low Entropy → Remove
    - Q4 (Calibration Data): Low PPL + High Entropy → Keep Full
    """

    def __init__(
        self,
        sample_keep_ratio: float = 0.5,
        token_keep_ratio: float = 0.7,
        neighbor_lambda: float = 0.5,
        bisect_max_iter: int = 10,
    ):
        """
        Args:
            sample_keep_ratio: Target ratio of samples to keep (Q2 + Q4)
            token_keep_ratio: Ratio of tokens to keep for Q2 samples
            neighbor_lambda: Smoothing coefficient for neighbor-aware token scoring
            bisect_max_iter: Maximum iterations for bisection search
        """
        self.sample_keep_ratio = sample_keep_ratio
        self.token_keep_ratio = token_keep_ratio
        self.neighbor_lambda = neighbor_lambda
        self.bisect_max_iter = bisect_max_iter

    def compute_ppl_and_entropy(
        self,
        model,
        tokens: torch.Tensor,
        response_start_idx: int,
    ) -> Tuple[float, float, List[float], List[float]]:
        """
        Compute sample-level and token-level PPL and Entropy.

        Args:
            model: The language model
            tokens: Token IDs [seq_len]
            response_start_idx: Index where response starts (prompt_length)

        Returns:
            Tuple of (sample_ppl, sample_entropy, token_ppls, token_entropies)
        """
        with torch.no_grad():
            # Forward pass
            outputs = model(tokens.unsqueeze(0), labels=tokens.unsqueeze(0))
            logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Compute token-level metrics for response tokens
            token_ppls = []
            token_entropies = []

            for i in range(response_start_idx, len(tokens)):
                # Get logits for predicting token i (using logits at position i-1)
                token_logits = logits[i - 1]
                log_probs = F.log_softmax(token_logits, dim=-1)
                probs = torch.exp(log_probs)

                # Token perplexity
                true_token_id = tokens[i]
                token_nll = -log_probs[true_token_id].item()
                token_ppl = np.exp(token_nll)
                token_ppls.append(token_ppl)

                # Token entropy
                entropy = -(probs * log_probs).sum().item()
                token_entropies.append(entropy)

            # Sample-level metrics (average over response tokens)
            sample_ppl = np.exp(np.mean([np.log(p) for p in token_ppls]))
            sample_entropy = np.mean(token_entropies)

            return sample_ppl, sample_entropy, token_ppls, token_entropies

    def bisect_search_thresholds(
        self,
        ppls: List[float],
        entropies: List[float],
    ) -> Tuple[float, float, float, float]:
        """
        Find optimal PPL and Entropy thresholds via bisection search.

        Args:
            ppls: List of sample perplexities
            entropies: List of sample entropies

        Returns:
            Tuple of (ppl_low, ppl_high, ent_low, ent_high)
        """
        ppls = np.array(ppls)
        entropies = np.array(entropies)

        alpha_low, alpha_high = 0.0, 0.49
        beta_low, beta_high = 0.0, 0.49

        for _ in range(self.bisect_max_iter):
            alpha = (alpha_low + alpha_high) / 2
            beta = (beta_low + beta_high) / 2

            # Compute thresholds from quantiles
            ppl_low = np.quantile(ppls, alpha)
            ppl_high = np.quantile(ppls, 1 - alpha)
            ent_low = np.quantile(entropies, beta)
            ent_high = np.quantile(entropies, 1 - beta)

            # Count samples in Q2 and Q4
            q2_q4_count = 0
            for ppl, ent in zip(ppls, entropies):
                quadrant = self._classify_quadrant(ppl, ent, ppl_low, ppl_high, ent_low, ent_high)
                if quadrant in ["Q2", "Q4"]:
                    q2_q4_count += 1

            ratio = q2_q4_count / len(ppls)

            # Adjust search range
            if ratio < self.sample_keep_ratio:
                # Too few samples kept, relax thresholds
                alpha_low = alpha
                beta_low = beta
            else:
                # Too many samples kept, tighten thresholds
                alpha_high = alpha
                beta_high = beta

        return ppl_low, ppl_high, ent_low, ent_high

    def _classify_quadrant(
        self,
        ppl: float,
        entropy: float,
        ppl_low: float,
        ppl_high: float,
        ent_low: float,
        ent_high: float,
    ) -> str:
        """
        Classify a sample into one of four quadrants.

        Returns:
            Quadrant label: "Q1", "Q2", "Q3", or "Q4"
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

        # Handle mid-range cases
        # High PPL (error) cases - treat as misconceptions or noise
        elif ppl_category == "high" and ent_category == "mid":
            return "Q2"  # Lean towards misconception

        # Low PPL (mastered) cases - treat as redundant or calibration
        elif ppl_category == "low" and ent_category == "mid":
            return "Q3"  # Lean towards redundant

        # Mid PPL cases - decide based on entropy
        elif ppl_category == "mid" and ent_category == "high":
            return "Q4"  # Uncertain but not extremely wrong
        elif ppl_category == "mid" and ent_category == "low":
            return "Q3"  # Somewhat redundant
        else:
            # (mid, mid) case - default to calibration
            return "Q4"

    def neighbor_aware_token_scoring(
        self,
        token_ppls: List[float],
    ) -> List[float]:
        """
        Compute neighbor-aware token scores.

        Score formula: s_i = (1-λ)*PPL_i + λ*(PPL_{i-1}+PPL_{i+1})/2

        Args:
            token_ppls: List of token perplexities

        Returns:
            List of token scores
        """
        scores = []
        for i in range(len(token_ppls)):
            ppl_i = token_ppls[i]
            ppl_prev = token_ppls[i - 1] if i > 0 else ppl_i
            ppl_next = token_ppls[i + 1] if i < len(token_ppls) - 1 else ppl_i

            score = (1 - self.neighbor_lambda) * ppl_i + \
                    self.neighbor_lambda * (ppl_prev + ppl_next) / 2
            scores.append(score)

        return scores

    def prune_tokens(
        self,
        tokens: torch.Tensor,
        token_ppls: List[float],
        response_start_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prune tokens based on neighbor-aware scoring.

        Args:
            tokens: Token IDs [seq_len]
            token_ppls: Token perplexities for response tokens
            response_start_idx: Index where response starts

        Returns:
            Tuple of (pruned_tokens, new_loss_mask)
        """
        # Compute scores
        scores = self.neighbor_aware_token_scoring(token_ppls)

        # Keep top-k tokens
        num_keep = max(1, int(len(scores) * self.token_keep_ratio))
        sorted_indices = np.argsort(scores)[:num_keep]  # Keep lowest scores (lowest PPL)
        sorted_indices = np.sort(sorted_indices)  # Restore order

        # Build pruned tokens and loss mask
        prompt_tokens = tokens[:response_start_idx]
        response_tokens = tokens[response_start_idx:]

        # Keep selected response tokens
        kept_response_tokens = response_tokens[sorted_indices]
        pruned_tokens = torch.cat([prompt_tokens, kept_response_tokens])

        # Build loss mask (0 for prompt, 1 for kept response tokens)
        loss_mask = torch.zeros(len(pruned_tokens), dtype=torch.long)
        loss_mask[response_start_idx:] = 1

        return pruned_tokens, loss_mask

    def prune_batch(
        self,
        model,
        rollout_data: Dict,
    ) -> Dict:
        """
        Apply Q-Tuning pruning to a batch of rollout data.

        This is the main entry point that implements Algorithm 1 from the paper.

        Args:
            model: The language model (for computing PPL and Entropy)
            rollout_data: Dictionary containing 'tokens', 'response_lengths', etc.

        Returns:
            Pruned rollout_data with updated 'tokens', 'loss_masks', etc.
        """
        tokens_list = rollout_data["tokens"]
        response_lengths = rollout_data["response_lengths"]

        # Stage 1: Compute PPL and Entropy for all samples
        sample_metrics = []
        for tokens, resp_len in zip(tokens_list, response_lengths):
            prompt_len = len(tokens) - resp_len
            ppl, ent, token_ppls, token_ents = self.compute_ppl_and_entropy(
                model, tokens, prompt_len
            )
            sample_metrics.append({
                "ppl": ppl,
                "entropy": ent,
                "token_ppls": token_ppls,
                "token_entropies": token_ents,
                "tokens": tokens,
                "response_start_idx": prompt_len,
            })

        # Find thresholds via bisection search
        ppls = [m["ppl"] for m in sample_metrics]
        entropies = [m["entropy"] for m in sample_metrics]
        ppl_low, ppl_high, ent_low, ent_high = self.bisect_search_thresholds(ppls, entropies)

        # Stage 2: Classify and prune
        kept_indices = []
        pruned_tokens_list = []
        pruned_loss_masks = []
        quadrant_counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}

        for idx, metrics in enumerate(sample_metrics):
            quadrant = self._classify_quadrant(
                metrics["ppl"], metrics["entropy"],
                ppl_low, ppl_high, ent_low, ent_high
            )
            quadrant_counts[quadrant] += 1

            # Keep Q2 and Q4 samples
            if quadrant in ["Q2", "Q4"]:
                kept_indices.append(idx)

                if quadrant == "Q2":
                    # Apply token pruning to Q2
                    pruned_tokens, loss_mask = self.prune_tokens(
                        metrics["tokens"],
                        metrics["token_ppls"],
                        metrics["response_start_idx"],
                    )
                    pruned_tokens_list.append(pruned_tokens)
                    pruned_loss_masks.append(loss_mask)
                else:
                    # Keep Q4 samples in full
                    tokens = metrics["tokens"]
                    loss_mask = torch.zeros(len(tokens), dtype=torch.long)
                    loss_mask[metrics["response_start_idx"]:] = 1
                    pruned_tokens_list.append(tokens)
                    pruned_loss_masks.append(loss_mask)

        # Build pruned rollout_data
        pruned_rollout_data = {}
        for key, val in rollout_data.items():
            if isinstance(val, list):
                if key == "tokens":
                    pruned_rollout_data[key] = pruned_tokens_list
                elif key == "loss_masks":
                    pruned_rollout_data[key] = pruned_loss_masks
                else:
                    # Keep other fields for kept samples
                    pruned_rollout_data[key] = [val[i] for i in kept_indices]
            else:
                pruned_rollout_data[key] = val

        # Update response_lengths and total_lengths
        if "response_lengths" in pruned_rollout_data:
            pruned_rollout_data["response_lengths"] = [
                len(tokens) - sample_metrics[i]["response_start_idx"]
                for i, tokens in zip(kept_indices, pruned_tokens_list)
            ]

        if "total_lengths" in pruned_rollout_data:
            pruned_rollout_data["total_lengths"] = [len(tokens) for tokens in pruned_tokens_list]

        # Log statistics
        print(f"[Q-Tuning] Quadrant distribution: {quadrant_counts}")
        print(f"[Q-Tuning] Kept {len(kept_indices)}/{len(tokens_list)} samples "
              f"({100 * len(kept_indices) / len(tokens_list):.1f}%)")

        return pruned_rollout_data
