# Q-Tuning Pruning Script - Usage Examples

## Quick Start

### 1. 测试模式 (原功能保留)
处理100个math样本 + 100个code样本（快速测试）

```bash
python tests/test_q_tuning_pruning.py
```

或者指定更少样本：
```bash
python tests/test_q_tuning_pruning.py --n-math 50 --n-code 50
```

### 2. 处理全部数据 ⭐ NEW!

```bash
python tests/test_q_tuning_pruning.py \
    --model-path /lustre/projects/polyullm/caishuo/cs_models/TL-1.5B-CPT-Base \
    --data-path /lustre/projects/polyullm/caishuo/cs_data/slime_sft/0726--57kmath_57kcode_34kscience_deduped--0.8-easy-math-code-final.json \
    --output-dir /lustre/projects/polyullm/caishuo/q_tuning_full_output \
    --n-math -1 \
    --n-code -1
```

**说明**：
- `--n-math -1` 表示处理**所有**math样本
- `--n-code -1` 表示处理**所有**code样本

### 3. 只处理全部math数据，code只取100个

```bash
python tests/test_q_tuning_pruning.py \
    --model-path /path/to/model \
    --data-path /path/to/data.json \
    --n-math -1 \
    --n-code 100
```

### 4. 调整pruning参数

```bash
python tests/test_q_tuning_pruning.py \
    --n-math -1 \
    --n-code -1 \
    --sample-keep-ratio 0.3 \     # 保留30%样本（更aggressive）
    --token-keep-ratio 0.5 \       # Q2样本只保留50%的token
    --neighbor-lambda 0.7          # 更重视相邻token的PPL
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `/Users/shuocai/Documents/code/iter_0010999__e8m0` | 模型路径 |
| `--data-path` | 数据集路径 | 输入数据JSON文件 |
| `--output-dir` | `./q_tuning_analysis_output` | 输出目录 |
| `--n-math` | `100` | Math样本数量，`-1`=全部 |
| `--n-code` | `100` | Code样本数量，`-1`=全部 |
| `--sample-keep-ratio` | `0.5` | Stage 1保留样本比例 |
| `--token-keep-ratio` | `0.7` | Stage 2 Q2样本保留token比例 |
| `--neighbor-lambda` | `0.5` | Token scoring中相邻token权重 |

## 支持的Category类型

脚本自动识别以下类别：

### Math样本
- `"math"`
- `"math-OT3"`
- `"Nemotron-math"`

### Code样本
- `"code-OT"`
- `"code-OT3"`
- `"Nemotron-code"`

**识别规则**：只要category字段**包含** `"math"` 或 `"code"` 关键词即可。

## 预期运行时间

### 服务器上 (CUDA GPU)

| 样本数 | 预计时间 |
|--------|----------|
| 200 (100+100) | 5-10分钟 |
| 1,000 | 25-50分钟 |
| 10,000 | 4-8小时 |
| 全部 (~72,000) | **约30-60小时** |

**建议**：
- 先用100+100测试确认pipeline正常
- 如果要处理全部数据，建议在后台运行：
  ```bash
  nohup python tests/test_q_tuning_pruning.py \
      --n-math -1 --n-code -1 \
      --model-path /path/to/model \
      --data-path /path/to/data.json \
      --output-dir /path/to/output \
      > q_tuning_full.log 2>&1 &
  ```

## 输出文件

处理完成后，在 `--output-dir` 中会生成：

```
q_tuning_analysis_output/
├── stage1_kept.json                        # Q2+Q4保留的样本
├── stage1_removed.json                     # Q1+Q3删除的样本
├── stage2_final.json                       # 最终样本（Q2已pruned tokens）
├── stage2_pruned_tokens_visualization.json # Token详细信息
├── token_pruning_visualization.html        # 🎨 可视化对比
└── summary_statistics.json                 # 统计摘要
```

### 检查统计信息

```bash
cat q_tuning_analysis_output/summary_statistics.json
```

示例输出：
```json
{
  "stage1": {
    "total_samples": 200,
    "Q1_count": 25,    // Harmful Noise - 删除
    "Q2_count": 60,    // Valuable Misconception - 保留+token pruning
    "Q3_count": 15,    // Redundant Knowledge - 删除
    "Q4_count": 100,   // Calibration Data - 完整保留
    "kept_count": 160,
    "actual_keep_ratio": 0.80
  },
  "stage2": {
    "q2_samples": 60,
    "q4_samples": 100,
    "total_tokens_before": 50000,
    "total_tokens_after": 40000,
    "token_compression_ratio": 0.80
  }
}
```

## 常见问题

### Q: 为什么处理全部数据需要这么久？
A: 每个样本需要：
- 模型forward pass计算PPL和Entropy
- 逐token计算perplexity
- 对于长样本，可能有几百上千个token

### Q: 可以分批处理吗？
A: 可以！比如：
```bash
# 批次1: 处理前10000个样本
python tests/test_q_tuning_pruning.py --n-math 5000 --n-code 5000 --output-dir batch1

# 批次2: 再处理10000个（需要修改代码支持offset）
# 目前脚本总是从头开始，建议一次处理完
```

### Q: 如何暂停和恢复？
A: 目前不支持断点续传。如果中断，需要重新运行。

### Q: 内存不够怎么办？
A:
1. 减少batch size（需要修改代码中的模型推理部分）
2. 使用更小的模型
3. 分批处理较少样本

## 使用建议

1. **先小规模测试** (100+100)
   - 验证pipeline正常
   - 检查pruning结果合理性
   - 调整 `sample_keep_ratio` 和 `token_keep_ratio`

2. **查看可视化结果**
   ```bash
   open q_tuning_analysis_output/token_pruning_visualization.html
   ```
   - 确认被删除的token确实是冗余的
   - 确认保留的token是核心推理步骤

3. **根据统计调整参数**
   - 如果Q1+Q3太多（>60%），说明数据质量问题或模型太好
   - 如果Q2太少（<20%），可能阈值设置不合理
   - 理想分布：Q1(10-20%), Q2(20-30%), Q3(10-20%), Q4(30-40%)

4. **全量处理**
   - 确认参数后，运行全量处理
   - 使用nohup在后台运行
   - 定期检查日志
