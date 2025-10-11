# Q-Tuning Pruning Analysis

This document explains the Q-Tuning two-stage pruning strategy and how to analyze the pruned data.

## Overview

Q-Tuning implements a two-stage data pruning approach based on the paper "Winning the Pruning Gamble" (arXiv:2509.23873):

1. **Stage 1: Sample-Level Pruning** - Removes entire samples based on Error-Uncertainty (EU) Plane
2. **Stage 2: Token-Level Pruning** - Selectively removes high-perplexity tokens from valuable misconceptions

## Key Differences Between Stages

### Stage 1: Sample-Level Pruning (EU Plane)

**What it does:** Classifies entire samples into 4 quadrants based on **Perplexity (PPL)** and **Entropy**:

| Quadrant | PPL | Entropy | Interpretation | Action |
|----------|-----|---------|----------------|--------|
| **Q1** | High | High | Harmful Noise - Model uncertain AND wrong | ❌ **REMOVED** |
| **Q2** | High | Low | Valuable Misconception - Model confident but wrong | ✅ **KEPT** → Token Pruning |
| **Q3** | Low | Low | Redundant Knowledge - Model already mastered | ❌ **REMOVED** |
| **Q4** | Low | High | Calibration Data - Model correct but uncertain | ✅ **KEPT** (full) |

**Implementation Details:**
- Uses bisection search to find PPL/Entropy thresholds that keep ~50% of samples (configurable)
- Removes **Q1** (noisy, harmful) and **Q3** (redundant, already learned)
- Keeps **Q2** (needs refinement via token pruning) and **Q4** (valuable calibration)

**Key Code:**
```python
# From q_tuning_pruner.py
def bisect_search_thresholds(self, ppls, entropies):
    # Find thresholds to keep sample_keep_ratio in Q2+Q4
    ppl_low, ppl_high = np.quantile(ppls, [alpha, 1-alpha])
    ent_low, ent_high = np.quantile(entropies, [beta, 1-beta])
```

### Stage 2: Token-Level Pruning (Q2 Only)

**What it does:** For **Q2 samples only**, removes high-perplexity tokens while keeping low-perplexity ones.

**Why Q2?** These samples have:
- **High PPL** (model makes errors) → Need refinement
- **Low Entropy** (model is confident) → Errors are systematic, not random

**Algorithm:**
1. Compute **neighbor-aware token scores** using surrounding context:
   ```
   score_i = (1-λ) × PPL_i + λ × (PPL_{i-1} + PPL_{i+1}) / 2
   ```
2. Keep tokens with **lowest scores** (lowest perplexity = easiest to predict)
3. Remove tokens with **highest scores** (highest perplexity = hardest to predict)

**Key Insight:** By removing high-PPL tokens, we focus training on the parts where the model is more confident, avoiding reinforcing systematic errors.

**Implementation Details:**
- Default `token_keep_ratio = 0.7` (keeps 70% of tokens)
- Uses neighbor smoothing (`neighbor_lambda = 0.5`) to avoid removing context
- **Q4 samples are kept in full** (no token pruning) as they provide valuable calibration

**Key Code:**
```python
# From q_tuning_pruner.py
def prune_tokens(self, tokens, token_ppls, response_start_idx):
    scores = self.neighbor_aware_token_scoring(token_ppls)
    num_keep = int(len(scores) * self.token_keep_ratio)
    sorted_indices = np.argsort(scores)[:num_keep]  # Keep lowest scores
```

## Usage

### Running the Analysis

```bash
python tests/test_q_tuning_pruning.py \
    --model-path /path/to/model \
    --data-path /path/to/data.json \
    --output-dir ./q_tuning_output \
    --n-math 100 \
    --n-code 100 \
    --sample-keep-ratio 0.5 \
    --token-keep-ratio 0.7
```

**Key Parameters:**
- `--n-math`: Number of math samples (set to `-1` for all)
- `--n-code`: Number of code samples (set to `-1` for all)
- `--sample-keep-ratio`: Target ratio for Q2+Q4 samples (default: 0.5)
- `--token-keep-ratio`: Ratio of tokens to keep in Q2 samples (default: 0.7)
- `--neighbor-lambda`: Neighbor smoothing weight (default: 0.5)
- `--ignore-special-tokens`: Ignore special tokens when computing PPL/Entropy (for Long CoT data)
- `--special-token-pairs`: Custom special token pairs (default: `<think>,</think>` and `<answer>,</answer>`)

### Output Files

1. **`stage1_kept.json`** - Samples kept after Stage 1 (Q2 + Q4)
2. **`stage1_removed.json`** - Samples removed in Stage 1 (Q1 + Q3)
3. **`stage2_final.json`** - Final training data after both stages
4. **`stage2_pruned_tokens_visualization.json`** - Token-level pruning details
5. **`token_pruning_visualization.html`** - Interactive HTML visualization
6. **`summary_statistics.json`** - Statistical summary

### Visualization

Open `token_pruning_visualization.html` to see:
- **Stage 1**: Sample distribution across Q1-Q4 quadrants with example previews
- **Stage 2**: Token-by-token visualization showing kept (green) vs removed (red) tokens
- **Statistics**: Overall compression ratios and sample counts

## Comparison: Stage 1 vs Stage 2

| Aspect | Stage 1 (Sample-Level) | Stage 2 (Token-Level) |
|--------|------------------------|----------------------|
| **Granularity** | Entire samples | Individual tokens |
| **Metric** | Sample PPL + Entropy | Token PPL + neighbor context |
| **Decision** | Keep/Remove whole sample | Keep/Remove specific tokens |
| **Applied to** | All samples | Q2 samples only |
| **Output** | Q2 + Q4 samples | Q2 (pruned) + Q4 (full) |
| **Goal** | Remove noise (Q1) and redundancy (Q3) | Refine misconceptions (Q2) |

## Example Workflow

```
Input: 200 samples (100 math + 100 code)
  ↓
Stage 1: Sample-Level Pruning
  • Q1 (Harmful Noise): 40 samples → REMOVED
  • Q2 (Valuable Misconception): 50 samples → KEPT (for token pruning)
  • Q3 (Redundant Knowledge): 60 samples → REMOVED
  • Q4 (Calibration Data): 50 samples → KEPT (full)
  ↓ 100 samples kept (50%)

Stage 2: Token-Level Pruning (Q2 only)
  • Q2: 50 samples × ~200 tokens/sample = 10,000 tokens
    → Keep 70% = 7,000 tokens (remove 3,000 high-PPL tokens)
  • Q4: 50 samples × ~200 tokens/sample = 10,000 tokens
    → Keep 100% = 10,000 tokens (no pruning)
  ↓
Final Output: 100 samples with 17,000 tokens total (85% compression)
```

## Key Insights

1. **Stage 1 removes samples entirely** - No recovery possible
   - Q1 samples are too noisy to be useful
   - Q3 samples are already learned (redundant)

2. **Stage 2 refines Q2 samples** - Keeps valuable structure while removing problematic tokens
   - Focuses on systematic misconceptions (confident errors)
   - Uses neighbor context to avoid breaking coherence

3. **Q4 samples are precious** - Never pruned at token level
   - Provide calibration for model uncertainty
   - Help model learn when to be uncertain

## Long CoT (Chain-of-Thought) Data Support

For Long CoT datasets where reasoning is wrapped in special tokens (e.g., `<think>...</think>` and `<answer>...</answer>`), these tokens often have **high perplexity** which can bias the pruning decisions.

### Problem

```
User: What is 2+2?
Assistant: <think>This is addition. 2+2=4.</think><answer>4</answer>
```

- `<think>` and `</think>` tokens have **high PPL** (model not trained on these markers)
- This can incorrectly classify good samples as Q1 (Harmful Noise)
- Token pruning might remove valuable reasoning steps

### Solution

Use `--ignore-special-tokens` to exclude these tokens from PPL/Entropy computation:

```bash
python tests/test_q_tuning_pruning.py \
    --model-path /path/to/model \
    --data-path /path/to/long_cot_data.json \
    --ignore-special-tokens \
    --special-token-pairs "<think>,</think>" "<answer>,</answer>"
```

### How It Works

The implementation uses **token-level matching** instead of text matching to handle tokenization properly:

1. **Pre-tokenizes special markers**: `<think>` → `[60, 27963, 62]` (e.g., `['<', 'think', '>']`)
2. **Pattern matching on token IDs**: Searches for exact token ID sequences in the response
3. **Identifies token ranges**: Marks all tokens between start and end patterns
4. **Stage 1 - Excludes from metrics**: Ignores marked tokens when computing sample-level PPL/Entropy
5. **Stage 2 - Force preservation**: Special tokens are **never pruned** during token-level pruning

**Key advantage**: Correctly handles cases where special markers are split across multiple tokens:
- `<think>` might tokenize as `['<', 'th', 'ink', '>']` (4 tokens)
- `</think>` might tokenize as `['</', 'th', 'ink', '>']` (4 tokens)
- All 8 tokens will be correctly identified and preserved

### Custom Special Tokens

You can specify any special token pairs:

```bash
--ignore-special-tokens \
--special-token-pairs \
    "<reasoning>,</reasoning>" \
    "<reflection>,</reflection>" \
    "<answer>,</answer>"
```

### Example Output

When running with `--ignore-special-tokens`, you'll see how special tokens are tokenized:

```bash
Special token tokenization preview:
  <think>              → [60, 27963, 62] = ['<', 'think', '>']
  </think>             → [1340, 27963, 62] = ['</', 'think', '>']
  <answer>             → [60, 12011, 62] = ['<', 'answer', '>']
  </answer>            → [1340, 12011, 62] = ['</', 'answer', '>']
```

**Without `--ignore-special-tokens`:**
```
Sample PPL: 45.2 (HIGH due to <think> tokens having high perplexity)
Quadrant: Q1 (Harmful Noise) → REMOVED ❌
```

**With `--ignore-special-tokens`:**
```
Sample PPL: 3.8 (computed only on actual reasoning, excluding special markers)
Quadrant: Q2 (Valuable Misconception) → KEPT ✅

Stage 2 Token Pruning for Q2 samples:
  Total tokens: 100
  Special tokens: 8 (<think>, </think>, <answer>, </answer>)
  Prunable tokens: 92
  Target keep ratio: 70%
  → Keep: 64 content tokens (70% of 92) + 8 special tokens = 72 tokens total
  → Remove: 28 content tokens only (special tokens preserved)
```

### When to Use

- ✅ Your data has special structural tokens (`<think>`, `<answer>`, etc.)
- ✅ These tokens weren't in the model's training data
- ✅ You want to focus on the content, not the markup
- ❌ Your data uses standard formats without special tokens
- ❌ Special tokens are part of your model's vocabulary

## References

- Paper: "Winning the Pruning Gamble: A Unified Approach to Joint Sample and Token Pruning" (arXiv:2509.23873)
- Implementation: `slime/utils/q_tuning_pruner.py`
- Analysis Script: `tests/test_q_tuning_pruning.py`
