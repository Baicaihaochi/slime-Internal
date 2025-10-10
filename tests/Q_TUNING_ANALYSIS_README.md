# Q-Tuning Data Pruning Analysis

This script implements the Q-Tuning pruning method from the paper:
**"Winning the Pruning Gamble: A Unified Approach to Joint Sample and Token Pruning for Efficient Supervised Fine-Tuning"**

## What It Does

The script analyzes your training data through two stages:

### Stage 1: Sample-Level Pruning
Classifies samples into 4 quadrants based on **Perplexity (PPL)** and **Entropy**:

| Quadrant | Characteristics | Action |
|----------|----------------|--------|
| **Q1: Harmful Noise** | High PPL + High Entropy | ❌ **REMOVE** - Unreliable/mislabeled |
| **Q2: Valuable Misconception** | High PPL + Low Entropy | ✅ **KEEP** + Token Pruning |
| **Q3: Redundant Knowledge** | Low PPL + Low Entropy | ❌ **REMOVE** - Already mastered |
| **Q4: Calibration Data** | Low PPL + High Entropy | ✅ **KEEP FULL** - Hard but reliable |

### Stage 2: Token-Level Pruning
For **Q2 samples only**, removes high-perplexity tokens using a **neighbor-aware scoring** mechanism:

```
token_score = (1-λ) × PPL_i + λ × (PPL_{i-1} + PPL_{i+1}) / 2
```

**Q4 samples** are kept completely intact to preserve calibration signals.

## Usage

### Quick Start

```bash
cd /Users/shuocai/Downloads/slime/tests
python test_q_tuning_pruning.py
```

### Configuration

Edit these parameters in the script's `main()` function:

```python
analyzer = QTuningAnalyzer(
    model_path="/Users/shuocai/Documents/code/iter_0010999__e8m0",  # Your model
    data_path="/Users/shuocai/Documents/code/cs_data/0726--57kmath_57kcode_34kscience_deduped--0.8-easy-math-code-final.json",
    output_dir="/Users/shuocai/Downloads/slime/tests/q_tuning_analysis_output",

    sample_keep_ratio=0.5,   # Keep 50% of samples (Q2 + Q4)
    token_keep_ratio=0.7,    # Keep 70% of tokens in Q2 samples
    neighbor_lambda=0.5,     # Neighbor weight in token scoring
)
```

### Requirements

```bash
pip install torch transformers tqdm numpy
```

## Output Files

After running, you'll find these files in `q_tuning_analysis_output/`:

### 📊 Main Results

1. **`stage1_kept.json`** - Samples retained after Stage 1 (Q2 + Q4)
   - Contains PPL, Entropy, and quadrant classification in `metadata`

2. **`stage1_removed.json`** - Samples removed in Stage 1 (Q1 + Q3)
   - Organized by quadrant: `{"Q1": [...], "Q3": [...]}`

3. **`stage2_final.json`** - Final samples after token pruning
   - Q2 samples have `token_mask` in metadata
   - Q4 samples marked as `"tokens_kept": "all"`

4. **`stage2_pruned_tokens_visualization.json`** - Token-level pruning details
   - Shows which tokens were kept/removed for each Q2 sample

5. **`token_pruning_visualization.html`** 🎨 **INTERACTIVE VISUALIZATION**
   - **Open this in your browser!**
   - Visual comparison of kept (green) vs removed (red) tokens
   - Hover over tokens to see their PPL scores
   - Shows first 50 Q2 samples

6. **`summary_statistics.json`** - Overall statistics
   ```json
   {
     "stage1": {
       "Q1_count": 25,
       "Q2_count": 60,
       "Q3_count": 15,
       "Q4_count": 40,
       "actual_keep_ratio": 0.50
     },
     "stage2": {
       "total_tokens_before": 15000,
       "total_tokens_after": 10500,
       "token_compression_ratio": 0.70
     }
   }
   ```

## Sample Metadata Structure

Each processed sample will have this metadata:

```json
{
  "id": 0,
  "problem": "...",
  "category": "math",
  "conversations": [...],
  "metadata": {
    "ppl": 8.65,                          // Sample-level perplexity
    "entropy": 1.54,                      // Sample-level entropy
    "token_ppls": [2.1, 15.3, 8.7, ...], // Per-token perplexity
    "token_entropies": [0.8, 1.2, ...],  // Per-token entropy
    "quadrant": "Q2",                     // Q1/Q2/Q3/Q4
    "token_mask": [1, 0, 1, 1, ...],     // 1=kept, 0=removed (Q2 only)
    "tokens_kept": 250,                   // Number of kept tokens
    "tokens_removed": 100                 // Number of removed tokens
  }
}
```

## Expected Runtime

- **Model loading**: ~30 seconds
- **Computing PPL/Entropy**: ~2-5 seconds per sample
- **Total for 200 samples**: ~15-20 minutes (depending on GPU)

## Analyzing Results

### 1. Check Statistics
```bash
cat q_tuning_analysis_output/summary_statistics.json
```

**What to look for:**
- Q2 (Misconception) should be **20-40%** of samples
- Q4 (Calibration) should be **20-40%** of samples
- Token compression in Q2 should match your `token_keep_ratio`

### 2. View Visualizations
```bash
open q_tuning_analysis_output/token_pruning_visualization.html
```

**What to look for:**
- Are removed tokens (red) actually noisy or redundant?
- Are kept tokens (green) the core reasoning steps?

### 3. Sample Q2 Examples
```bash
jq '.[] | select(.metadata.quadrant == "Q2") | {id, ppl, entropy, tokens_removed}' q_tuning_analysis_output/stage2_final.json | head -20
```

### 4. Sample Q4 Examples (for comparison)
```bash
jq '.[] | select(.metadata.quadrant == "Q4") | {id, ppl, entropy}' q_tuning_analysis_output/stage1_kept.json | head -20
```

## Troubleshooting

### Error: "Cannot load model"
- Check that model path exists: `ls /Users/shuocai/Documents/code/iter_0010999__e8m0`
- Ensure model is in HuggingFace format (not Megatron torch_dist)

### Error: "Out of memory"
- Reduce batch size in model inference
- Process fewer samples: Change `n_math=50, n_code=50` in `load_samples()`

### Warning: "Not enough math/code samples"
- Your dataset might not have clear category labels
- Check the `category` field in your data

### All samples classified as Q1 or Q3
- Your model might be too good or too bad on this data
- Try adjusting `sample_keep_ratio` to 0.3 or 0.7

## Integration with slime Training

Once you've validated the pruning strategy works well:

1. **Use pruned data for training:**
   ```bash
   # Use stage2_final.json as your training data
   cp q_tuning_analysis_output/stage2_final.json /path/to/training/data.json
   ```

2. **Implement dynamic pruning in slime:**
   - Add PPL/Entropy computation to `slime/backends/megatron_utils/loss.py`
   - Apply sample filtering per epoch
   - Apply token masking via `loss_mask`

3. **Expected improvements:**
   - 30-40% speedup (fewer samples + fewer tokens)
   - Similar or **better** performance (removes noise)
   - More stable training (Q4 calibration samples)

## Paper Reference

Wang et al. (2025). "Winning the Pruning Gamble: A Unified Approach to Joint Sample and Token Pruning for Efficient Supervised Fine-Tuning"

Key insights:
- **First method to consistently outperform full-data training**
- SmolLM2-1.7B: +38% improvement with only 12.5% data
- LLaMA3-8B on GSM8K: 48.07 with 35% data (vs 42.08 full-data)

## Questions?

If the results look suspicious:
1. Check `summary_statistics.json` - are quadrant distributions reasonable?
2. Open the HTML visualization - do removed tokens make sense?
3. Sample a few examples from each quadrant manually
4. Try different `sample_keep_ratio` values (0.3, 0.5, 0.7)
