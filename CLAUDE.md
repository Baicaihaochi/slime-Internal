# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**slime** is an LLM post-training framework for RL scaling that connects Megatron-LM with SGLang to enable high-performance distributed reinforcement learning training (PPO/GRPO). It supports training from 4B to 355B+ parameter models with various parallelism strategies.

## Essential Commands

### Environment Setup

```bash
# Install slime in development mode
pip install -e .

# Install pre-commit hooks for code style
apt install pre-commit -y
pre-commit install
```

### Model Checkpoint Conversion

```bash
# Convert HuggingFace → Megatron torch_dist format
cd /root/slime
source scripts/models/glm4-9B.sh  # Load model config
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/hf_model \
    --save /path/to/torch_dist_output

# Convert Megatron → HuggingFace format
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/torch_dist_ckpt/iter_xxx/ \
  --output-dir /path/to/hf_output \
  --origin-hf-dir /path/to/original_hf_model
```

### Training

```bash
# Single-node training (synchronous)
bash scripts/run-qwen3-4B.sh

# Single-node training (asynchronous, higher throughput)
python train_async.py [args...]

# Multi-node training via Ray cluster
# On head node:
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats
# On worker nodes:
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8

# Submit training job:
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}' \
   -- python3 train.py [args...]
```

### Testing

```bash
# Run tests
pytest tests/

# Quick start test (GLM4-9B example)
bash tests/test_quick_start_glm4-9B.sh

# Test specific model configurations
bash tests/test-qwen2.5-0.5B-gsm8k.sh
```

### Documentation

```bash
# Build documentation
cd docs && bash build.sh

# Serve documentation locally
cd docs && bash serve.sh
```

## Architecture Overview

### Core Components

slime follows a **producer-consumer architecture** with three main subsystems:

1. **Training Backend** ([slime/backends/](slime/backends/))
   - **Megatron integration** ([slime/backends/megatron_utils/](slime/backends/megatron_utils/)): Primary training engine with TP/PP/EP/CP support
   - **Actor model** ([actor.py](slime/backends/megatron_utils/actor.py)): Manages training loop, log prob computation, advantage estimation
   - **Weight synchronization** ([update_weight_utils.py](slime/backends/megatron_utils/update_weight_utils.py)): IPC-based (colocate) or NCCL-based weight updates
   - **Loss functions** ([loss.py](slime/backends/megatron_utils/loss.py)): PPO/GRPO loss with KL penalty
   - Also supports FSDP and XTuner backends

2. **Rollout System** ([slime/rollout/](slime/rollout/))
   - **SGLang integration** ([sglang_rollout.py](slime/rollout/sglang_rollout.py)): Asynchronous generation engine
   - **Reward models** ([rm_hub/](slime/rollout/rm_hub/)): Built-in reward models (math, dapo, deepscaler, f1)
   - **Filters** ([filter_hub/](slime/rollout/filter_hub/)): Dynamic sampling filters (e.g., reward variance checks)
   - Supports custom generation functions for multi-turn dialogues and tool calling

3. **Ray Orchestration** ([slime/ray/](slime/ray/))
   - **Placement groups** ([placement_group.py](slime/ray/placement_group.py)): GPU allocation with PACK strategy for locality
   - **Actor group** ([actor_group.py](slime/ray/actor_group.py)): Distributed training coordinator
   - **Rollout manager** ([rollout.py](slime/ray/rollout.py)): Inference engine coordinator with sglang-router
   - **Data buffer** ([buffer.py](slime/ray/buffer.py)): Central coordinator for data flow and reward processing

### Training Workflow

**Data Flow:**
```
Prompt Dataset → RolloutManager (SGLang) → Generated Samples →
RolloutController (Buffer) → Training Data → ActorModel (Megatron) →
Weight Update → Rollout Engines → [Repeat]
```

**Two Training Modes:**

1. **Synchronous** ([train.py](train.py)):
   - Sequential: generate → train → update weights
   - Supports GPU memory offloading (`--offload`)
   - Required for colocation mode (`--colocate`)

2. **Asynchronous** ([train_async.py](train_async.py)):
   - Pipelines generation and training for 30-40% higher throughput
   - Overlaps next rollout generation with current training
   - Batched weight updates (`--update-weights-interval`)
   - No offloading support

### Plugin System

slime uses **function path arguments** for extensive customization:

- `--rollout-function-path`: Custom rollout generator (default: [sglang_rollout.py:generate_rollout](slime/rollout/sglang_rollout.py))
- `--custom-generate-function-path`: Custom generation logic for multi-turn/tool calling
- `--custom-rm-path`: Custom reward model (see [rm_hub/](slime/rollout/rm_hub/) for examples)
- `--custom-loss-function-path`: Custom training loss
- `--dynamic-sampling-filter-path`: Filter sample groups during generation
- `--buffer-filter-path`: Custom buffer sampling strategy
- `--custom-reward-post-process-path`: Custom advantage computation
- `--rollout-data-postprocess-path`: Pre-training data processing
- `--custom-megatron-init-path`: Custom Megatron initialization
- `--custom-megatron-before-log-prob-hook-path`: Pre-forward hook
- `--custom-megatron-before-train-step-hook-path`: Pre-training step hook

See [examples/](examples/) for implementation patterns.

## Key Implementation Details

### Weight Update Mechanism

Two modes based on `--colocate`:

1. **IPC Mode (Colocation)**: Training and inference share GPUs
   - Uses `torch.distributed.gather_object` for serialized tensors
   - Converts Megatron sharded weights → HuggingFace format → SGLang
   - Memory-efficient but requires careful `--sglang-mem-fraction-static` tuning

2. **NCCL Mode (Separate GPUs)**: Dedicated training and inference GPUs
   - Uses `torch.distributed.broadcast` via NCCL process groups
   - Pauses generation during weight sync
   - Higher throughput, more GPU memory required

Implementation: [update_weight_utils.py](slime/backends/megatron_utils/update_weight_utils.py)

### SGLang Integration

- SGLang servers launched as Ray actors ([sglang_engine.py](slime/backends/sglang_utils/sglang_engine.py))
- HTTP-based communication via sglang-router for load balancing
- All SGLang parameters accessible with `--sglang-` prefix (e.g., `--sglang-mem-fraction-static`)
- Router can be external (`--sglang-router-ip`, `--sglang-router-port`) for custom workflows

### Megatron Integration

- Requires Megatron in `PYTHONPATH` (e.g., `export PYTHONPATH=/root/Megatron-LM`)
- Imports parameters from `megatron.training.arguments.parse_args`
- Model configs in [scripts/models/](scripts/models/) define architecture hyperparameters
- Checkpoint format: `torch_dist` (recommended, auto-sharding) or `torch` (legacy)
- Checkpoint structure: `/path/iter_XXXXXX/*.distcp` + `latest_checkpointed_iteration.txt`

### Data Format

JSONL format with configurable keys:

```jsonl
{"prompt": [{"role": "user", "content": "..."}], "label": "...", "metadata": {...}}
```

Configured via:
- `--input-key prompt` (maps to Sample.prompt)
- `--label-key label` (maps to Sample.label)
- `--metadata-key metadata` (maps to Sample.metadata, useful for custom functions)
- `--apply-chat-template` (applies HuggingFace chat template)

### Sample Object

Core data structure ([types.py:Sample](slime/utils/types.py)):

- `tokens`: Full token sequence (prompt + response)
- `response_length`: Number of tokens in response
- `loss_mask`: Per-token training mask (1 = train, 0 = mask)
- `reward`: Scalar reward or dict for multi-objective
- `rollout_log_probs`: For importance sampling
- `status`: COMPLETED | TRUNCATED | ABORTED
- `metadata`: Custom data passed from dataset

### Parallelism Configuration

Configure in training scripts (see [scripts/models/](scripts/models/) for examples):

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 2        # TP
   --sequence-parallel                   # Megatron SP (always enable with TP)
   --pipeline-model-parallel-size 1      # PP
   --context-parallel-size 2             # CP (ring attention)
   --expert-model-parallel-size 1        # EP (for MoE)
   --expert-tensor-parallel-size 1       # ETP (TP for experts)

   # Recomputation for memory efficiency
   --recompute-granularity full          # or "selective"
   --recompute-method uniform
   --recompute-num-layers 1

   # Dynamic batching (recommended)
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
)
```

### Advanced Features

**Dynamic Sampling:**
- Over-sample prompts (`--over-sampling-batch-size > --rollout-batch-size`)
- Filter groups with `--dynamic-sampling-filter-path`
- Example: [check_reward_nonzero_std](slime/rollout/filter_hub/dynamic_sampling_filters.py) ensures reward variance

**Partial Rollout:**
- Recycle aborted samples with `--partial-rollout`
- Custom buffer strategy via `--buffer-filter-path`

**Multi-Turn/Agent Training:**
- Use `--custom-generate-function-path` for multi-step interaction loops
- Set `loss_mask=0` for tool outputs, `loss_mask=1` for model actions
- Store context in `sample.metadata` (pass via `--metadata-key`)

**FP8 Inference with BF16 Training:**
- Use FP8 HuggingFace checkpoint for `--hf-checkpoint`
- Keep BF16 Megatron checkpoint for `--ref-load` and `--load`

**Debugging:**
- `--save-debug-rollout-data`: Persist rollout samples
- `--load-debug-rollout-data`: Replay rollouts without inference
- `--debug-train-only`: Skip rollout, train on saved data
- `--debug-rollout-only`: Skip training, test generation

## Argument Categories

Arguments are divided into three categories:

1. **Megatron arguments**: Read from `PYTHONPATH` Megatron installation (e.g., `--tensor-model-parallel-size`)
2. **SGLang arguments**: Prefix with `--sglang-` (e.g., `--sglang-mem-fraction-static`)
3. **slime arguments**: Defined in [slime/utils/arguments.py](slime/utils/arguments.py)

See [docs/en/get_started/usage.md](docs/en/get_started/usage.md) for complete argument descriptions.

## Common Development Tasks

### Adding a Custom Reward Model

1. Create reward function in [slime/rollout/rm_hub/](slime/rollout/rm_hub/) or custom file:
```python
async def my_reward(args, sample: Sample, **kwargs) -> float:
    # Compute reward from sample.response and sample.label
    return score
```

2. Register in training script:
```bash
--custom-rm-path path.to.module:my_reward
```

### Adding a Custom Generation Function

1. Create async generation function:
```python
async def my_generate(args, sample: Sample, sampling_params) -> Sample:
    # Multi-turn loop
    sample.response = "..."
    sample.tokens = [...]
    sample.response_length = len(response_tokens)
    sample.loss_mask = [1, 1, 0, 0, ...]  # 1=train, 0=mask
    return sample
```

2. Configure:
```bash
--custom-generate-function-path path.to.module:my_generate
```

### Adding a New Model Architecture

1. Create config in [scripts/models/](scripts/models/):
```bash
MODEL_ARGS=(
   --num-layers X
   --hidden-size Y
   # ... other arch params
)
```

2. If not in Megatron's supported architectures, add config mapping in [slime/backends/megatron_utils/config_mapping/](slime/backends/megatron_utils/config_mapping/)

3. Register in [registry.py](slime/backends/megatron_utils/config_mapping/registry.py)

### Extending for New Backends

slime supports multiple training backends via [slime/backends/](slime/backends/):

- **Megatron** (primary): [megatron_utils/](slime/backends/megatron_utils/)
- **FSDP**: [fsdp_utils/](slime/backends/fsdp_utils/)
- **XTuner**: [xtuner_utils/](slime/backends/xtuner_utils/)

To add a new backend, implement the actor interface from [actor.py](slime/backends/megatron_utils/actor.py).

## Code Style

- **Formatting**: Black (line length 119) + isort
- **Linting**: Ruff (line length 119)
- **Pre-commit hooks**: Auto-format on commit
- Install: `pre-commit install`

Configuration: [pyproject.toml](pyproject.toml)

## Important Files Reference

- **Main entry points**: [train.py](train.py), [train_async.py](train_async.py)
- **Arguments**: [slime/utils/arguments.py](slime/utils/arguments.py)
- **Training loop**: [slime/backends/megatron_utils/actor.py](slime/backends/megatron_utils/actor.py)
- **Loss computation**: [slime/backends/megatron_utils/loss.py](slime/backends/megatron_utils/loss.py)
- **Generation**: [slime/rollout/sglang_rollout.py](slime/rollout/sglang_rollout.py)
- **Weight updates**: [slime/backends/megatron_utils/update_weight_utils.py](slime/backends/megatron_utils/update_weight_utils.py)
- **Resource allocation**: [slime/ray/placement_group.py](slime/ray/placement_group.py)
- **Data types**: [slime/utils/types.py](slime/utils/types.py)

## Documentation

- **Quick Start**: [docs/en/get_started/quick_start.md](docs/en/get_started/quick_start.md)
- **Usage Guide**: [docs/en/get_started/usage.md](docs/en/get_started/usage.md)
- **Debugging**: [docs/en/developer_guide/debug.md](docs/en/developer_guide/debug.md)
- **Blog**: [slime: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/)
- **Examples**: [examples/](examples/) (fully_async, multi_agent, search-r1, retool)

## Additional Resources

- **Model configs**: [scripts/models/](scripts/models/) contains configs for Qwen, GLM, LLaMA, DeepSeek, etc.
- **Training scripts**: [scripts/run-*.sh](scripts/) for various models and sizes
- **Plugins**: [slime_plugins/](slime_plugins/) for model-specific logic and extensions
- **Tests**: [tests/](tests/) for integration tests and examples
