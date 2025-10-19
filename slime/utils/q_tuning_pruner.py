"""
Q-Tuning: Dynamic Data Pruning for Efficient LLM Fine-Tuning

This module implements the Q-Tuning algorithm from "Winning the Pruning Gamble" (arXiv:2509.23873).
Q-Tuning performs joint sample and token pruning based on the Error-Uncertainty (EU) Plane, which
categorizes training data into four quadrants using perplexity (error) and entropy (uncertainty).

Reference: https://arxiv.org/abs/2509.23873
"""

import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from slime.utils.ppo_utils import calculate_log_probs_and_entropy


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
            model: The language model (can be a single model or a list for Megatron PP)
            tokens: Token IDs [seq_len]
            response_start_idx: Index where response starts (prompt_length)

        Returns:
            Tuple of (sample_ppl, sample_entropy, token_ppls, token_entropies)
        """
        # Handle Megatron model list (for Pipeline Parallelism)
        if isinstance(model, list):
            # Use the first model in the list (they all share the same forward logic)
            model = model[0]

        with torch.no_grad():
            # Store original tokens and seq_len (DO NOT modify the input parameter!)
            original_tokens = tokens
            seq_len = tokens.size(0)

            # Get tensor parallel size (required for Sequence Parallelism padding)
            try:
                from megatron.core import parallel_state as mpu

                tp_size = mpu.get_tensor_model_parallel_world_size()
                tp_group = mpu.get_tensor_model_parallel_group()
            except Exception:
                tp_size = 1
                tp_group = None

            # For Sequence Parallelism: BOTH batch_size and seq_len must be divisible by TP size
            # Pad sequence length if needed
            padded_seq_len = seq_len
            if seq_len % tp_size != 0:
                padded_seq_len = ((seq_len + tp_size - 1) // tp_size) * tp_size

            # Create padded tokens (DO NOT modify original tokens!)
            if padded_seq_len > seq_len:
                pad_length = padded_seq_len - seq_len
                # Pad with zeros (or model's pad_token_id if available)
                padded_tokens = torch.cat([original_tokens, torch.zeros(pad_length, dtype=original_tokens.dtype, device=original_tokens.device)])
            else:
                padded_tokens = original_tokens

            # Ensure batch_size is also divisible by TP size
            batch_size = max(tp_size, 1)
            batch_tokens = padded_tokens.unsqueeze(0).expand(batch_size, -1)  # [batch_size, padded_seq_len]

            # Create position_ids: [batch_size, padded_seq_len]
            position_ids = torch.arange(padded_seq_len, dtype=torch.long, device=original_tokens.device).unsqueeze(0).expand(batch_size, -1)

            # Create attention_mask: [batch_size, 1, padded_seq_len, padded_seq_len]
            # For padded tokens, mask them out in attention
            attention_mask = torch.tril(
                torch.ones((padded_seq_len, padded_seq_len), dtype=torch.bool, device=original_tokens.device)
            )
            # Mask out padded positions
            attention_mask[seq_len:, :] = False
            attention_mask[:, seq_len:] = False
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

            # Forward pass with padded inputs
            # Megatron models return logits directly as a tensor, not wrapped in an object
            outputs = model(
                input_ids=batch_tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=None,  # We don't need loss computation
            )

            # Extract logits from the first sample, only keep original sequence length
            # outputs is a tensor of shape [batch_size, padded_seq_len, vocab_size]
            logits = outputs[0, :seq_len, :]  # [seq_len, vocab_size]

            # Compute token-level metrics for response tokens
            # IMPORTANT: Use seq_len (not len(tokens)) to avoid accessing padded tokens
            eval_indices = list(range(response_start_idx, seq_len))
            if not eval_indices:
                return 0.0, 0.0, [], []

            token_ppls: List[float] = []
            token_entropies: List[float] = []

            use_vocab_parallel_ops = False
            if tp_group is not None:
                dist_module = getattr(torch, "distributed", None)
                if dist_module is not None:
                    try:
                        use_vocab_parallel_ops = dist_module.is_available() and dist_module.is_initialized()
                    except RuntimeError:
                        use_vocab_parallel_ops = False

            if use_vocab_parallel_ops:
                logits_indices = []
                for idx in eval_indices:
                    prev_idx = idx - 1
                    # Skip the first token if prev_idx < 0 (can't predict first token from nothing)
                    if prev_idx < 0:
                        continue
                    logits_indices.append(prev_idx)

                # If no valid indices, return default values
                if not logits_indices:
                    return 1.0, 0.0, [], []

                # Update eval_indices to match (skip first token if needed)
                valid_eval_indices = [idx for idx in eval_indices if idx > 0]

                logits_for_targets = logits[logits_indices].contiguous()
                target_tokens = original_tokens[valid_eval_indices].contiguous()

                log_probs_tensor, entropy_tensor = calculate_log_probs_and_entropy(
                    logits_for_targets,
                    target_tokens,
                    tp_group,
                    with_entropy=True,
                )

                log_probs_tensor = torch.atleast_1d(log_probs_tensor.squeeze(-1))
                entropy_tensor = torch.atleast_1d(entropy_tensor.squeeze(-1))
                token_nlls = -log_probs_tensor

                # Clamp to avoid numerical issues
                token_nlls = torch.clamp(token_nlls, min=0.0, max=50.0)
                token_ppls_tensor = token_nlls.exp()

                sample_ppl = float(token_nlls.mean().exp().item())
                sample_entropy = float(entropy_tensor.mean().item())
                token_ppls = [float(v) for v in token_ppls_tensor.cpu().tolist()]
                token_entropies = [float(v) for v in entropy_tensor.cpu().tolist()]
            else:
                for idx in eval_indices:
                    prev_idx = idx - 1
                    # Skip the first token if prev_idx < 0 (can't predict first token from nothing)
                    if prev_idx < 0:
                        continue

                    token_logits = logits[prev_idx]
                    log_probs = F.log_softmax(token_logits, dim=-1)
                    probs = torch.exp(log_probs)

                    true_token_id = original_tokens[idx]
                    token_nll = -log_probs[true_token_id].item()
                    # Clamp to avoid numerical explosion
                    token_nll = np.clip(token_nll, 0.0, 50.0)
                    token_ppl = np.exp(token_nll)
                    token_ppls.append(token_ppl)

                    entropy = -(probs * log_probs).sum().item()
                    token_entropies.append(entropy)

                # If no tokens were processed, return defaults
                if not token_ppls:
                    return 1.0, 0.0, [], []

                # Use mean of log(ppl) for numerical stability
                sample_ppl = np.exp(np.mean([np.log(max(p, 1e-10)) for p in token_ppls]))
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
        base_loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prune tokens based on neighbor-aware scoring.

        Args:
            tokens: Token IDs [seq_len]
            token_ppls: Token perplexities for response tokens
            response_start_idx: Index where response starts

        Returns:
            Loss mask tensor for response tokens only (length = response_len).
        """
        response_tokens = tokens[response_start_idx:]
        response_len = response_tokens.size(0)

        if len(token_ppls) == 0 or response_len == 0:
            if base_loss_mask is not None:
                if isinstance(base_loss_mask, torch.Tensor):
                    base_mask = base_loss_mask.clone().detach()
                else:
                    base_mask = torch.tensor(base_loss_mask, dtype=torch.long, device=tokens.device)
                if base_mask.size(0) == tokens.size(0):
                    base_mask = base_mask[response_start_idx: response_start_idx + response_len]
                return base_mask.to(device=tokens.device, dtype=torch.long)
            return torch.zeros(response_len, dtype=torch.long, device=tokens.device)

        scores = self.neighbor_aware_token_scoring(token_ppls)

        num_keep = len(scores)
        if self.token_keep_ratio < 1.0:
            # keep highest priority tokens (lowest score)
            num_keep = max(1, math.ceil(len(scores) * self.token_keep_ratio))
        sorted_indices = np.argsort(scores)[:num_keep]
        sorted_indices = np.sort(sorted_indices)

        if base_loss_mask is not None:
            if isinstance(base_loss_mask, torch.Tensor):
                base_mask = base_loss_mask.clone().detach()
            else:
                base_mask = torch.tensor(base_loss_mask, dtype=torch.long, device=tokens.device)
        else:
            base_mask = torch.ones(response_len, dtype=torch.long, device=tokens.device)

        if base_mask.dim() != 1:
            raise ValueError(f"Expected 1D loss mask, got shape {base_mask.shape}")

        if base_mask.size(0) not in (response_len, tokens.size(0)):
            raise ValueError(
                f"Loss mask length {base_mask.size(0)} incompatible with response length {response_len}"
            )

        if base_mask.size(0) == tokens.size(0):
            # convert full-length mask to response-only mask
            base_mask = base_mask[response_start_idx: response_start_idx + response_len]
        else:
            base_mask = base_mask.to(device=tokens.device, dtype=torch.long)

        kept_indices_tensor = torch.from_numpy(sorted_indices).long().to(base_mask.device)
        if kept_indices_tensor.numel() == response_len:
            new_mask = base_mask.clone()
        else:
            new_mask = torch.zeros_like(base_mask)
            new_mask[kept_indices_tensor] = base_mask[kept_indices_tensor]

        if new_mask.sum() == 0:
            print("[Q-Tuning Warning] All tokens masked out; forcing one token to remain for stability.")
            fallback_idx = int(kept_indices_tensor[0].item()) if kept_indices_tensor.numel() > 0 else 0
            fallback_idx = max(0, min(fallback_idx, response_len - 1))
            new_mask[fallback_idx] = 1

        return new_mask.to(device=tokens.device, dtype=torch.long)

    @staticmethod
    def _normalize_values(values: List[float]) -> List[float]:
        arr = np.array(values, dtype=np.float32)
        if arr.size == 0:
            return []
        v_min = float(arr.min())
        v_max = float(arr.max())
        if abs(v_max - v_min) < 1e-6:
            return [0.5 for _ in values]
        return [float((v - v_min) / (v_max - v_min)) for v in values]

    def _target_keep_count(self, total: int) -> int:
        if total == 0:
            return 0
        if self.sample_keep_ratio <= 0.0:
            return 0
        if self.sample_keep_ratio >= 1.0:
            return total
        keep = math.ceil(self.sample_keep_ratio * total)
        return max(1, min(total, keep))

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
        loss_masks_list = rollout_data.get("loss_masks")
        total_lengths_list = rollout_data.get("total_lengths")

        num_samples = len(tokens_list)
        if num_samples == 0:
            return None

        # Stage 1: Compute PPL and Entropy for all samples
        sample_metrics: List[Dict] = []
        for idx, (tokens, resp_len) in enumerate(zip(tokens_list, response_lengths)):
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
                "original_response_length": resp_len,
                "loss_mask": loss_masks_list[idx] if loss_masks_list is not None else None,
                "total_length": total_lengths_list[idx] if total_lengths_list is not None else len(tokens),
            })

        # Find thresholds via bisection search
        ppls = [m["ppl"] for m in sample_metrics]
        entropies = [m["entropy"] for m in sample_metrics]
        ppl_low, ppl_high, ent_low, ent_high = self.bisect_search_thresholds(ppls, entropies)

        norm_ppls = self._normalize_values(ppls)
        norm_ents = self._normalize_values(entropies)

        # Stage 2: Classify and prune
        kept_indices: List[int] = []
        pruned_tokens_list = []
        pruned_loss_masks = []
        quadrant_counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}

        for idx, metrics in enumerate(sample_metrics):
            quadrant = self._classify_quadrant(
                metrics["ppl"], metrics["entropy"],
                ppl_low, ppl_high, ent_low, ent_high
            )
            quadrant_counts[quadrant] += 1
            metrics["quadrant"] = quadrant
            metrics["norm_ppl"] = norm_ppls[idx]
            metrics["norm_entropy"] = norm_ents[idx]
            metrics["support_score"] = abs(norm_ppls[idx] - norm_ents[idx])
            if quadrant == "Q2":
                metrics["keep_priority"] = norm_ppls[idx] - norm_ents[idx]
            elif quadrant == "Q4":
                metrics["keep_priority"] = norm_ents[idx] - norm_ppls[idx]
            else:
                metrics["keep_priority"] = -metrics["support_score"]

        target_keep = self._target_keep_count(num_samples)
        if target_keep == 0:
            print("[Q-Tuning] Sample keep ratio requested 0; skipping batch.")
            return None

        primary_indices = [i for i, m in enumerate(sample_metrics) if m["quadrant"] in {"Q2", "Q4"}]
        primary_sorted = sorted(primary_indices, key=lambda i: sample_metrics[i]["keep_priority"], reverse=True)
        kept_indices = primary_sorted[:target_keep]

        if len(kept_indices) < target_keep:
            fallback_candidates = [
                i for i, m in enumerate(sample_metrics)
                if m["quadrant"] in {"Q1", "Q3"} and i not in kept_indices
            ]
            fallback_sorted = sorted(fallback_candidates, key=lambda i: sample_metrics[i]["support_score"], reverse=True)
            for cand in fallback_sorted:
                if len(kept_indices) >= target_keep:
                    break
                kept_indices.append(cand)

        if len(kept_indices) < target_keep:
            # as a last resort, add remaining samples by descending support
            remaining = [
                i for i in range(num_samples)
                if i not in kept_indices
            ]
            remaining_sorted = sorted(remaining, key=lambda i: sample_metrics[i]["support_score"], reverse=True)
            for cand in remaining_sorted:
                if len(kept_indices) >= target_keep:
                    break
                kept_indices.append(cand)

        if not kept_indices:
            print("[Q-Tuning] No samples selected after pruning; keeping the best scoring sample to avoid stall.")
            fallback_idx = int(np.argmax([m["support_score"] for m in sample_metrics]))
            kept_indices = [fallback_idx]

        # Ensure deterministic order
        kept_indices = sorted(kept_indices)

        for idx in kept_indices:
            metrics = sample_metrics[idx]
            quadrant = metrics["quadrant"]
            tokens = metrics["tokens"]
            base_loss_mask = metrics["loss_mask"]
            response_start_idx = metrics["response_start_idx"]

            if quadrant == "Q2":
                loss_mask = self.prune_tokens(
                    tokens,
                    metrics["token_ppls"],
                    response_start_idx,
                    base_loss_mask=base_loss_mask,
                )
            else:
                response_length = len(tokens) - response_start_idx
                if base_loss_mask is not None:
                    if isinstance(base_loss_mask, torch.Tensor):
                        loss_mask = base_loss_mask.clone().detach()
                    else:
                        loss_mask = torch.tensor(base_loss_mask, dtype=torch.long, device=tokens.device)
                    if loss_mask.dim() != 1:
                        raise ValueError(f"Expected 1D loss mask, got shape {loss_mask.shape}")
                    if loss_mask.size(0) == tokens.size(0):
                        loss_mask = loss_mask[response_start_idx: response_start_idx + response_length]
                    loss_mask = loss_mask.to(device=tokens.device, dtype=torch.long)
                else:
                    loss_mask = torch.ones(response_length, dtype=torch.long, device=tokens.device)

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
                sample_metrics[i]["original_response_length"] for i in kept_indices
            ]

        if "total_lengths" in pruned_rollout_data:
            pruned_rollout_data["total_lengths"] = [sample_metrics[i]["total_length"] for i in kept_indices]

        quadrant_id_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        pruned_rollout_data["q_tuning_quadrant"] = [quadrant_id_map[sample_metrics[i]["quadrant"]] for i in kept_indices]

        # Log statistics
        print(f"[Q-Tuning] Quadrant distribution: {quadrant_counts}")
        print(f"[Q-Tuning] Kept {len(kept_indices)}/{len(tokens_list)} samples "
              f"({100 * len(kept_indices) / len(tokens_list):.1f}%)")

        return pruned_rollout_data
