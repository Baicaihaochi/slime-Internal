# High-Entropy Token Filtering for RLVR

基于论文 "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning" 的实现。

## 原理

论文发现在Chain-of-Thought推理中：
- 只有少数token（约20%）具有高熵值，这些token作为"分叉点"（forking tokens）决定推理方向
- 多数token（约80%）具有低熵值，只是沿着已确定的路径执行
- **仅在高熵token上应用policy gradient更新**即可达到与全token训练相当甚至更好的性能

核心发现：
- 在Qwen3-8B上：使用top 20%高熵token = 性能与100% token相当
- 在Qwen3-14B上：+4.79 on AIME'25, +5.21 on AIME'24
- 在Qwen3-32B上：+11.04 on AIME'25, +7.71 on AIME'24
- **越大的模型，效果越显著**

## 使用方法

### 启用高熵token过滤

在训练脚本中添加以下参数：

```bash
python train.py \
    --high-entropy-token-filter \
    --entropy-percentile 0.2 \
    [其他参数...]
```

### 参数说明

- `--high-entropy-token-filter`: 启用高熵token过滤（默认关闭）
- `--entropy-percentile`: 保留的高熵token百分比（默认0.2，即20%）
  - 0.2 = 只对top 20%高熵token计算梯度
  - 0.1 = 只对top 10%高熵token计算梯度（更激进，可能损失性能）
  - 0.5 = 只对top 50%高熵token计算梯度（较保守）

### 完整示例

```bash
#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-32B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-32B
   --ref-load /root/Qwen3-32B_torch_dist
   --load /root/Qwen3-32B_slime/
   --save /root/Qwen3-32B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --num-steps-per-rollout 1
   --global-batch-size 128
   --rollout-max-response-len 8192
   --rollout-temperature 0.8
   --balance-data
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # 启用高熵token过滤
   --high-entropy-token-filter
   --entropy-percentile 0.2
)

# 其他参数...
python train.py \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    # ...
```

## 实现细节

实现非常简洁优雅，只需在[slime/backends/megatron_utils/loss.py](slime/backends/megatron_utils/loss.py)中：

1. 计算所有token的entropy
2. 根据`entropy_percentile`计算阈值（只从有效token中计算）
3. 创建高熵token mask
4. 将其与原loss_mask相乘，只保留高熵token
5. 重新计算`sum_of_sample_mean`

核心代码约60行，无破坏性修改，完全兼容现有代码。

## 论文关键发现

### 1. CoT中的熵模式
- 80th percentile entropy ≈ 0.672
- 高熵token示例："however", "wait", "thus", "suppose", "given"（逻辑连接词）
- 低熵token示例：代码片段、数学表达式、单词后缀（高确定性）

### 2. RLVR训练中的熵演化
- RLVR保留base model的熵模式（86%+ overlap）
- 主要调整高熵token的熵值
- 低熵token熵值变化很小

### 3. 最佳比例
- 20% 效果最佳（论文Figure 7）
- 10% 移除了部分有用token，削弱探索
- 50%/100% 加入低熵token，降低探索效率

### 4. 泛化能力
- 在数学数据集训练，在LiveCodeBench（代码）上测试仍然优于全token训练
- 说明高熵token与模型泛化能力相关

## 理论解释（Discussion）

### 为什么RL泛化而SFT记忆？
- RL保留或增加高熵token的熵 → 保持推理路径灵活性
- SFT将输出推向one-hot分布 → 降低高熵token熵 → 失去推理路径灵活性

### 为什么LLM CoT与传统RL不同？
- 传统RL：所有action entropy均匀分布
- LLM CoT：混合低熵majority + 高熵minority
- 原因：预训练知识 + 可读性要求 → 大部分token必须符合语言结构（低熵）

### 为什么clip-higher优于entropy bonus？
- Entropy bonus均匀增加所有token熵 → 破坏低熵majority
- Clip-higher（ε_high=0.28）只增加高importance ratio token的熵
- 高importance ratio token往往是高熵token → 精准作用

## 适用场景

✅ **推荐使用：**
- 大模型（≥14B）RLVR训练
- 数学推理、代码生成等需要长CoT的任务
- 计算资源有限，希望提升训练效率

⚠️ **谨慎使用：**
- 小模型（<8B）可能因容量不足，效果不明显
- 非推理任务（如对话、翻译）可能不适用

❌ **不建议：**
- SFT训练（论文未验证）

## 性能对比

| Model | Baseline (All Tokens) | Forking Tokens (20%) | Improvement |
|-------|----------------------|---------------------|-------------|
| Qwen3-8B | 33.33 (AIME'24) | 34.58 | +1.25 |
| Qwen3-14B | 45.21 (AIME'24) | 50.42 | **+5.21** |
| Qwen3-32B | 55.83 (AIME'24) | 63.54 | **+7.71** |
| Qwen3-32B | 45.63 (AIME'25) | 56.67 | **+11.04** |

论文Table 2原始数据。

## 引用

```bibtex
@article{wang2025beyond,
  title={Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning},
  author={Wang, Shenzhi and Yu, Le and Gao, Chang and Zheng, Chujie and Liu, Shixuan and Lu, Rui and others},
  journal={arXiv preprint arXiv:2506.01939},
  year={2025}
}
```

## 论文链接

- arXiv: https://arxiv.org/abs/2506.01939
- Project Page: https://shenzhi-wang.github.io/high-entropy-minority-tokens-rlvr
