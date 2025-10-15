#!/bin/bash

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# ================== model config =============================
SCRIPT_DIR=/root/slime/scripts
source "${SCRIPT_DIR}/models/qwen2.5-7B.sh"
# =============================================================

# ================= user config ===============================

LOG_DIR=/home/projects/polyullm/kejing/slime_workspace/wandb/


LOAD_DIR=/home/projects/polyullm/kejing/slime_workspace/slime_polaris/ckpt/Polaris-7B-bf16TI--8kstg1
SAVE_DIR=/home/projects/polyullm/kejing/slime_workspace/slime_polaris/ckpt/Polaris-7B-bf16TI--8kstg1
POLARIS_TRACKING_DIR=/home/projects/polyullm/kejing/slime_workspace/slime_polaris/Polaris-7B-bf16TI--8kstg1/polaris_reward_tracking

DATA_DIR=/lustre/projects/polyullm/caishuo/cs_data/slime_rl/polaris-data-53K.jsonl

# HF_CHECKPOINT=/lustre/projects/polyullm/caishuo/slime/InfiAlign-SFT-Qwen-7B-165K
HF_CHECKPOINT=/lustre/projects/polyullm/models/Qwen/Qwen2.5-7B-Instruct
#/lustre/projects/polyullm/caishuo/slime-0907/models/Qwen2.5-Math-7B__slime__fp8-bsz64-w0.05-f32scale-stg2-165k--0912----hf/iter_0012909__f32
REF_LOAD=/home/projects/polyullm/caishuo/cs20251004/cs_models/slime_sft_models/TL0920_stg2/
# ==============================================================

# ================ paralle config ==============================
TP=4
PP=1
CP=2
EP_MP=1
EP_TP=1
MAX_TOKENS_PER_GPU=8192
# ==============================================================

# ================ RL specific config =========================
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz)) #$((train_prompt_bsz * 3))

NUM_ROLLOUT=10240
N_SAMPLES_PER_PROMPT=16
GLOBAL_BATCH_SIZE=2048
ROLLOUT_MAX_RESPONSE_LEN=8192
ROLLOUT_TEMPERATURE=1.0   #1.1
OVER_SAMPLING_BATCH_SIZE=${gen_prompt_bsz}
# ==============================================================

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   --load ${LOAD_DIR}
   --save ${SAVE_DIR}
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}
   --input-key prompt
   --label-key label
   --apply-chat-template
   # --rollout-shuffle  # remove shufffle
   --rm-type deepscaler
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${train_prompt_bsz}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-temperature ${ROLLOUT_TEMPERATURE}
   --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}  # ${gen_prompt_bsz}
   # --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size ${GLOBAL_BATCH_SIZE}  # ${train_prompt_bsz}
   --balance-data


)

POLARIS_ARGS=(
    --enable-polaris-dynamic-sampling
    --polaris-good-reward-min 0.1
    --polaris-good-reward-max 0.9
    --polaris-min-good-ratio 0.33
    --enable-polaris-reward-tracking
    --polaris-reward-tracking-dir ${POLARIS_TRACKING_DIR}
    --polaris-verbose
   # align with verl's behavior
    --polaris-skip-batch-when-insufficient
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /home/projects/polyullm/kejing/slime_workspace/data/aime-2024.jsonl
   --n-samples-per-eval-prompt 1 # 16
   --eval-max-response-len 30000
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP}
   --context-parallel-size ${CP}
   --expert-model-parallel-size ${EP_MP}
   --expert-tensor-parallel-size ${EP_TP}

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group Polaris-7B-8kstg1
   --wandb-mode offline
   --wandb-dir ${LOG_DIR}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.5
   --sglang-max-running-requests 128

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

PRECISE_ARGS=(
   --transformer-impl transformer_engine
   --bf16
   #--fp8-format e4m3
   #--fp8-recipe blockwise
   #--fp8-param-gather
   # --direct-update-fp8-weight
)

TENSORBOARD_ARGS=(
   --profile-step-start 10
   --profile-step-end 12
   --tensorboard-dir ${LOG_DIR}/tensorboard
   --record-memory-history
)

# launch the master node of ray in container
export http_proxy=""
export https_proxy=""




# Build the runtime environment JSON with proper variable substitution
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
        "working_dir": "/home/projects/polyullm/kejing/slime_workspace/slime_polaris/slime",
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/:/home/projects/polyullm/kejing/slime_workspace/slime_polaris/slime",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:1024",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
            "NVTE_DEBUG": "0",
            "HOME": "/home/projects/polyullm/kejing/slime_workspace/slime_polaris/slime",
            "http_proxy": "",
            "https_proxy": "",
            "NCCL_SOCKET_IFNAME": "bond0",
            "WANDB_MODE": "offline",
            "WANDB_DIR": "/home/projects/polyullm/kejing/slime_workspace/wandb/",
            "RAY_DEDUP_LOGS_ALLOW_REGEX": "kejing",
            "NO_PROXY": "localhost,127.0.0.1,klb-dgx-*",
            "no_proxy": "localhost,127.0.0.1,klb-dgx-*"
        }
    }' \
   -- python3 train.py \
   --actor-num-nodes 4 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${POLARIS_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${PRECISE_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   $@

# you can add more args to the script
# bash ./run-InfiAlign-SFT-Qwen-7B-165K-2node-rl.sh --num-rollout 20480 --rollout-batch-size 256
# even in a sbatch script:
# sbatch --nodes=2 submit_4node_rl.sh ./run-InfiAlign-SFT-Qwen-7B-165K-2node-rl.sh --actor-num-nodes 2


