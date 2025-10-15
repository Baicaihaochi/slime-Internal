#!/bin/bash
# Example training script with POLARIS features enabled
#
# This demonstrates how to use POLARIS dynamic sampling and reward tracking
# in SLIME RL training.

set -e

# Configuration
MODEL_PATH="/lustre/projects/polyullm/models/Qwen/Qwen2.5-7B-Instruct"
DATA_PATH="/lustre/projects/polyullm/caishuo/cs_data/slime_rl/polaris-data-53K.jsonl"
EXPERIMENT_NAME="polaris_example"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"
TRACKING_DIR="${OUTPUT_DIR}/reward_tracking"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${TRACKING_DIR}

echo "=================================================="
echo "POLARIS-enabled SLIME Training Example"
echo "=================================================="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Tracking: ${TRACKING_DIR}"
echo "=================================================="

# Training with POLARIS features
PYTHONPATH=/root/Megatron-LM:/lustre/projects/polyullm/caishuo/slime1012/slime python train.py \
    --hf-checkpoint ${MODEL_PATH} \
    --rollout-data-path ${DATA_PATH} \
    \
    `# Cluster configuration` \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 8 \
    --rollout-num-gpus-per-engine 1 \
    --colocate \
    \
    `# Training configuration` \
    --global-batch-size 128 \
    --rollout-batch-size 128 \
    --n-samples-per-prompt 4 \
    --num-epoch 10 \
    --use-hf-config-for-megatron \
    \
    `# POLARIS features - Dynamic Sampling` \
    --enable-polaris-dynamic-sampling \
    --polaris-good-reward-min 0.0 \
    --polaris-good-reward-max 1.0 \
    --polaris-min-good-ratio 0.33 \
    \
    `# POLARIS features - Reward Tracking` \
    --enable-polaris-reward-tracking \
    --polaris-reward-tracking-dir ${TRACKING_DIR} \
    \
    `# Verbose logging` \
    --polaris-verbose \
    \
    `# Rollout configuration` \
    --rollout-temperature 1.0 \
    --rollout-top-p 1.0 \
    --rollout-top-k -1 \
    --rollout-max-response-len 2048 \
    \
    `# Algorithm configuration` \
    --advantage-estimator grpo \
    --use-kl-loss \
    --kl-loss-coef 0.001 \
    --kl-loss-type low_var_kl \
    \
    `# Optimizer configuration` \
    --lr 1e-6 \
    --min-lr 1e-7 \
    --lr-decay-style cosine \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    \
    `# Checkpointing` \
    --save ${OUTPUT_DIR}/checkpoints \
    --save-interval 100 \
    --load ${OUTPUT_DIR}/checkpoints \
    \
    `# Logging` \
    --use-wandb \
    --wandb-name ${EXPERIMENT_NAME} \
    --wandb-project "slime-polaris" \
    \
    `# Other` \
    --seed 42

echo "=================================================="
echo "Training complete!"
echo "Reward tracking log: ${TRACKING_DIR}/${EXPERIMENT_NAME}.jsonl"
echo "=================================================="
