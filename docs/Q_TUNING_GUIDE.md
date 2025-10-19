# Q-Tuning: Dynamic Data Pruning for Efficient LLM Fine-Tuning

## Overview

Q-Tuning is a dynamic data pruning method that implements joint sample and token pruning based on the **Error-Uncertainty (EU) Plane** framework. It categorizes training data into four quadrants using perplexity (model error) and entropy (model uncertainty), then applies targeted pruning strategies.

**Reference**: [Winning the Pruning Gamble (arXiv:2509.23873)](https://arxiv.org/abs/2509.23873)

## Key Concepts

### Error-Uncertainty (EU) Plane

The EU Plane maps each training sample onto a 2D space:
- **X-axis (Error)**: Perplexity (PPL) - How surprising the ground truth is to the model
- **Y-axis (Uncertainty)**: Entropy - How uncertain the model's predictions are

### Four Quadrants

1. **Q1 (Harmful Noise)**: High PPL + High Entropy
   - Unreliable or mislabeled data
   - **Action**: Remove via sample pruning

2. **Q2 (Valuable Misconception)**: High PPL + Low Entropy
   - Confidently wrong responses with correctable errors
   - **Action**: Keep + Apply token-level pruning to isolate core misconceptions

3. **Q3 (Redundant Knowledge)**: Low PPL + Low Entropy
   - Already mastered content with low marginal gain
   - **Action**: Remove via sample pruning

4. **Q4 (Calibration Data)**: Low PPL + High Entropy
   - Hard but reliable samples essential for confidence calibration
   - **Action**: Keep in full (no token pruning)

## Usage

### Enable Q-Tuning

Add the following arguments to your training script:

```bash
--enable-q-tuning \
--q-tuning-sample-keep-ratio 0.5 \
--q-tuning-token-keep-ratio 0.7 \
--q-tuning-neighbor-lambda 0.5 \
--q-tuning-bisect-max-iter 10
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-q-tuning` | flag | False | Enable Q-Tuning dynamic data pruning |
| `--q-tuning-sample-keep-ratio` | float | 0.5 | Target ratio of samples to keep (Q2 + Q4) |
| `--q-tuning-token-keep-ratio` | float | 0.7 | Ratio of tokens to keep for Q2 samples |
| `--q-tuning-neighbor-lambda` | float | 0.5 | Smoothing coefficient for neighbor-aware token scoring (0-1) |
| `--q-tuning-bisect-max-iter` | int | 10 | Maximum iterations for bisection search |

### Example: Training with Q-Tuning

```bash
# Single-node training with Q-Tuning (25% sample + 70% token retention)
bash scripts/run-qwen3-4B.sh \
    --enable-q-tuning \
    --q-tuning-sample-keep-ratio 0.25 \
    --q-tuning-token-keep-ratio 0.7

# Multi-node training with Q-Tuning (50% sample + 50% token retention)
python train.py \
    --enable-q-tuning \
    --q-tuning-sample-keep-ratio 0.5 \
    --q-tuning-token-keep-ratio 0.5 \
    --q-tuning-neighbor-lambda 0.5 \
    --global-batch-size 256 \
    --num-rollout 1000
```

## Implementation Details

### Two-Stage Pruning Process

#### Stage 1: Sample-Level Pruning (EU Plane Construction)

1. **Compute Metrics**: For each sample in the mini-batch:
   - Calculate sample-level perplexity: `PPL = exp(mean(token_NLLs))`
   - Calculate sample-level entropy: `Ent = mean(token_entropies)`

2. **Find Thresholds**: Use bisection search to find quantile-based thresholds (α*, β*) such that:
   - `ppl_low = Quantile_α(PPL)`
   - `ppl_high = Quantile_{1-α}(PPL)`
   - `ent_low = Quantile_β(Ent)`
   - `ent_high = Quantile_{1-β}(Ent)`
   - These thresholds are chosen so that `|Q2 ∪ Q4| / |batch| ≈ sample_keep_ratio`

3. **Classify & Prune**:
   - Assign each sample to Q1, Q2, Q3, or Q4 based on thresholds
   - Remove Q1 and Q3 samples entirely

#### Stage 2: Token-Level Pruning (Q2 Only)

1. **Neighbor-Aware Scoring**: For each token i in Q2 samples:
   ```python
   score_i = (1-λ) * PPL_i + λ * (PPL_{i-1} + PPL_{i+1}) / 2
   ```
   - This smoothing avoids removing isolated high-PPL tokens that may be semantically important

2. **Keep Top-k Tokens**: Rank tokens by score and keep the top `token_keep_ratio` fraction

3. **Preserve Q4 Samples**: Keep all tokens in Q4 samples (no token pruning)

### Dynamic Per-Batch Operation

**Key Feature**: Q-Tuning recomputes PPL and Entropy at **each training step** using the **current model state** (fθ_t), not a fixed initial model.

- **Why?**: As training progresses, the model's understanding evolves. A sample that was "Harmful Noise" (Q1) early on might become "Calibration Data" (Q4) later.
- **Performance**: Uses gradient-free forward passes, adding ~10-20% overhead per batch.

## Expected Results

Based on the paper (SmolLM2-1.7B, WizardLM dataset):

| Configuration | Avg Performance | Data Used | Speedup |
|---------------|-----------------|-----------|---------|
| Full Data SFT | 30.58 | 100% | 1.0x |
| Q-Tuning (12.5% sample, 50% token) | **37.74** | 6.25% | ~16x |
| Q-Tuning (25% sample, 70% token) | **36.87** | 17.5% | ~5.7x |
| Random Pruning (same budget) | 33.98 | 6.25% | ~16x |

**Key Insight**: Q-Tuning is the first dynamic pruning method to consistently outperform full-data training.

## Hyperparameter Sensitivity

### Sample Keep Ratio
- **0.5 (default)**: Balanced performance, 2x speedup
- **0.25**: Higher efficiency, may sacrifice some performance
- **0.75**: Conservative, closer to full-data performance

### Token Keep Ratio
- **0.7 (default)**: Recommended for most tasks
- **0.5**: More aggressive, higher risk
- **0.9**: Conservative, minimal token pruning

### Neighbor Lambda (λ)
- **0.5 (default)**: Balanced smoothing
- **0.0**: No smoothing (pure PPL-based pruning)
- **0.7-1.0**: More aggressive smoothing (use for noisy data)

### Ablation Study Results (from paper)

| Method | λ | GSM8K | SQuAD | TriviaQA | Avg |
|--------|---|-------|-------|----------|-----|
| PPL (λ=0) | 0.0 | 25.32 | 29.71 | 56.54 | 45.92 |
| **Q-Tuning (λ=0.5)** | 0.5 | **26.08** | **32.79** | **56.17** | **46.79** |
| Reversed PPL | 0.5 | 16.68 | 32.01 | 55.47 | 44.86 |

## Debugging & Monitoring

### Enable Verbose Logging

Q-Tuning automatically prints statistics at each training step:

```
[Q-Tuning] Quadrant distribution: {'Q1': 142, 'Q2': 89, 'Q3': 251, 'Q4': 518}
[Q-Tuning] Kept 607/1000 samples (60.7%)
```

### Visualize EU Plane

You can add custom logging to visualize the EU Plane distribution:

```python
# In your custom hook (--rollout-data-postprocess-path)
def visualize_eu_plane(args, rollout_data):
    ppls = rollout_data.get("sample_ppls", [])
    entropies = rollout_data.get("sample_entropies", [])

    import matplotlib.pyplot as plt
    plt.scatter(ppls, entropies, alpha=0.5)
    plt.xlabel("Perplexity")
    plt.ylabel("Entropy")
    plt.savefig(f"eu_plane_step_{args.rollout_id}.png")
```

## Compatibility

### Supported Features
- ✅ Megatron backend (primary)
- ✅ Tensor Parallelism (TP)
- ✅ Pipeline Parallelism (PP)
- ✅ Context Parallelism (CP)
- ✅ Dynamic batch sizing (`--use-dynamic-batch-size`)
- ✅ Offloading (`--offload`)
- ✅ Colocated training/inference (`--colocate`)

### Not Yet Supported
- ❌ FSDP backend (requires adaptation)
- ❌ XTuner backend (requires adaptation)
- ❌ Multi-turn dialogue pruning (future work)

## Advanced Usage

### Combine with Other Features

```bash
# Q-Tuning + Dynamic Batching + Offloading
python train.py \
    --enable-q-tuning \
    --q-tuning-sample-keep-ratio 0.5 \
    --q-tuning-token-keep-ratio 0.7 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 4608 \
    --offload
```

### Custom Quadrant Logic

If you need custom quadrant classification, you can modify `q_tuning_pruner.py`:

```python
# In QTuningPruner._classify_quadrant()
# Example: Be more conservative with Q1 (Harmful Noise)
if ppl_category == "high" and ent_category == "high":
    # Only remove if PPL is VERY high
    if ppl > ppl_high * 1.5:
        return "Q1"
    else:
        return "Q2"  # Treat as misconception instead
```

## Troubleshooting

### Issue: "Out of Memory during Q-Tuning"

**Solution**: Q-Tuning requires forward passes for all samples. Reduce `--rollout-batch-size` or increase `--rollout-num-gpus`.

### Issue: "Too many/few samples kept"

**Solution**: Adjust `--q-tuning-sample-keep-ratio`. The bisection search should converge to the target ratio within 10 iterations.

### Issue: "Performance degradation"

**Possible causes**:
1. `token_keep_ratio` too low (try 0.7-0.8)
2. Dataset has unusual PPL/Entropy distribution
3. Model is undertrained (Q-Tuning works best with somewhat trained models)

## Citations

If you use Q-Tuning in your research, please cite:

```bibtex
@article{wang2025qtuning,
  title={Winning the Pruning Gamble: A Unified Approach to Joint Sample and Token Pruning for Efficient Supervised Fine-Tuning},
  author={Wang, Shaobo and Wang, Jiaming and Zhang, Jiajun and ...},
  journal={arXiv preprint arXiv:2509.23873},
  year={2025}
}
```

## See Also

- [Dynamic Sampling Filters](../examples/): Custom filtering strategies
- [Custom Loss Functions](../docs/en/developer_guide/custom_loss.md): Integrate with custom training objectives
- [Debugging Guide](../docs/en/developer_guide/debug.md): Debug Q-Tuning behavior
