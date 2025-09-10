#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
source "${SCRIPT_DIR}/models/qwen3-4B-Thinking-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B-Thinking-2507/
   --ref-load /root/Qwen3-4B-Thinking-2507_torch_dist/
   --load /root/Qwen3-4B-Thinking-2507_slime/
   --save /root/Qwen3-4B-Thinking-2507_slime/
   --save-interval 1000
)

SFT_ARGS=(
   --rollout-function-path sft_rollout.generate_rollout
   --prompt-data /path/to/your/data.json
   --input-key messages
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 64
   --global-batch-size 64

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   #--recompute-granularity full
   #--recompute-method uniform
   #--recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
#    --lr-warmup-iters 1349
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.9
   --weight-decay 0
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-4B-thinking-2507-sft-fp8
   --wandb-key ${WANDB_KEY}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

PERCISE_ARGS=(
   --transformer-impl transformer_engine
   --bf16
   --fp8-format e4m3
   --fp8-recipe blockwise
   --fp8-param-gather
   #--fp8-recipe mxfp8
   #--reuse-grad-buf-for-mxfp8-param-ag
   #--activation-func-fp8-input-store
   #--overlap-grad-reduce
   #--overlap-param-gather
)

TENSORBOARD_ARGS=(
   --use-pytorch-profiler
   --profile-step-start 16
   --profile-step-end 18
   --tensorboard-dir /root/tensorboard/qwen3-4b-thinking-sft-fp8
   --record-memory-history
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD\": \"1\",
    \"TORCH_NCCL_AVOID_RECORD_STREAMS\": \"1\",
    \"NVTE_ALLOW_NONDETERMINISTIC_ALGO\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"NVTE_DEBUG\": \"0\",
    \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${PERCISE_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   ${MISC_ARGS[@]}