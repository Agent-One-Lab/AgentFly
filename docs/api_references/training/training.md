# Training Guide

## Overview

AgentFly uses reinforcement learning (RL) to train agents, specifically leveraging Proximal Policy Optimization (PPO) with various advantage estimation methods. The training system is built on top of VERL (Volcano Engine Reinforcement Learning) and supports distributed training with Ray.

## Training Command

The main training entry point is:

```bash
python -m agentfly.cli train [OPTIONS]
```

All configuration is done through Hydra command-line overrides, allowing flexible configuration without modifying code.

## Training Options

### Agent Configuration

Configure the agent behavior and capabilities:

- **`agent.use_agent`**: Enable agent mode (default: `True`)
- **`agent.init_args.model_name_or_path`**: HuggingFace model identifier or path (e.g., `Qwen/Qwen2.5-3B-Instruct`)
- **`agent.init_args.template`**: Chat template name (e.g., `qwen2.5`, `qwen2.5-vl`)
- **`agent.init_args.agent_type`**: Agent type - `hf`, `react`, `code`, `gui`, etc.
- **`agent.init_args.tools`**: List of tools available to the agent (e.g., `[calculator]`, `[google_search,answer]`)
- **`agent.init_args.reward_name`**: Name of the reward function to use (e.g., `math_reward_string_equal`, `qa_f1_reward`)
- **`agent.init_args.backend`**: Backend for agent execution - `async_verl` (recommended) or others
- **`agent.max_turns`**: Maximum number of interaction turns per episode
- **`agent.num_chains`**: Number of parallel interaction chains per sample
- **`agent.max_tokens_per_turn`**: Max tokens to generate per turn.

### Algorithm Configuration

Configure the RL algorithm:

- **`algorithm.adv_estimator`**: Advantage estimation method:
    - `gae`: Generalized Advantage Estimation (default)
    - `grpo`: Group Relative Policy Optimization
    - `reinforce_plus_plus`: REINFORCE++ estimator
    - `rloo`: REINFORCE Leave-One-Out
    - `remax`: REINFORCE with Max
- **`algorithm.gamma`**: Discount factor for future rewards (default: `1.0`)
- **`algorithm.lam`**: GAE lambda parameter (default: `1.0`)
- **`algorithm.use_kl_in_reward`**: Whether to include KL penalty in reward (default: `False`)
- **`algorithm.kl_penalty`**: KL estimation method - `kl`, `abs`, `mse`, `low_var_kl`, or `full`
- **`algorithm.kl_ctrl.type`**: KL control type - `fixed` or `adaptive`
- **`algorithm.kl_ctrl.kl_coef`**: KL penalty coefficient (default: `0.001`)
- **`algorithm.kl_ctrl.target_kl`**: Target KL divergence for adaptive control (default: `0.1`)
- **`algorithm.kl_ctrl.horizon`**: Horizon for adaptive controller (default: `10000`)

### Trainer Configuration

Configure training execution:

- **`trainer.project_name`**: Project name for experiment tracking (e.g., `AgentRL`)
- **`trainer.experiment_name`**: Experiment name for run identification
- **`trainer.logger`**: Logging backends - `['console']`, `['wandb']`, or `['console','wandb']`
- **`trainer.total_training_steps`**: Total number of training steps
- **`trainer.total_epochs`**: Total number of training epochs (alternative to steps)
- **`trainer.nnodes`**: Number of nodes for distributed training (default: `1`)
- **`trainer.n_gpus_per_node`**: Number of GPUs per node (default: `8`)
- **`trainer.save_freq`**: Frequency (in iterations) to save checkpoints (default: `-1` for no saving)
- **`trainer.test_freq`**: Frequency (in iterations) to run validation (default: `-1` for no validation)
- **`trainer.val_before_train`**: Whether to run validation before training starts (default: `True`)
- **`trainer.critic_warmup`**: Number of iterations to warm up critic before policy updates (default: `0`)
- **`trainer.resume_mode`**: Resume mode - `auto`, `disable`, or `resume_path`
- **`trainer.resume_from_path`**: Path to resume from (when `resume_mode=resume_path`)

### Data Configuration

Configure training and validation data:

- **`data.train_files`**: Path to training dataset file (JSON format)
- **`data.val_files`**: Path to validation dataset file (JSON format)
- **`data.train_batch_size`**: Training batch size
- **`data.val_batch_size`**: Validation batch size

### Actor/Rollout/Reference Configuration

Configure the actor model, rollout engine, and reference model:

- **`actor_rollout_ref.actor.optim.lr`**: Learning rate for actor (e.g., `5e-7`)
- **`actor_rollout_ref.actor.ppo_mini_batch_size`**: PPO mini-batch size
- **`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`**: Micro-batch size per GPU
- **`actor_rollout_ref.actor.use_kl_loss`**: Whether to use KL loss in actor training (default: `True`)
- **`actor_rollout_ref.actor.kl_loss_coef`**: KL loss coefficient
- **`actor_rollout_ref.actor.kl_loss_type`**: KL loss type - `mse`, `kl`, etc.
- **`actor_rollout_ref.actor.entropy_coeff`**: Entropy coefficient for exploration (default: `0.001`)
- **`actor_rollout_ref.actor.fsdp_config.param_offload`**: Enable parameter offloading for FSDP
- **`actor_rollout_ref.actor.fsdp_config.optimizer_offload`**: Enable optimizer offloading for FSDP
- **`actor_rollout_ref.model.path`**: Model path for actor
- **`actor_rollout_ref.model.enable_gradient_checkpointing`**: Enable gradient checkpointing (default: `False`)
- **`actor_rollout_ref.rollout.name`**: Rollout engine name - `vllm` (recommended)
- **`actor_rollout_ref.rollout.response_length`**: Maximum response length for rollouts
- **`actor_rollout_ref.rollout.tensor_model_parallel_size`**: Tensor parallelism size for rollout
- **`actor_rollout_ref.rollout.gpu_memory_utilization`**: GPU memory utilization for vLLM (default: `0.5`)
- **`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`**: Micro-batch size for log probability computation
- **`actor_rollout_ref.ref.fsdp_config.param_offload`**: Parameter offloading for reference model

### Critic Configuration

Configure the value function (critic):

- **`critic.model.path`**: Model path for critic
- **`critic.ppo_mini_batch_size`**: PPO mini-batch size for critic
- **`critic.ppo_micro_batch_size_per_gpu`**: Micro-batch size per GPU for critic

## Example Training Commands

### Basic Math Agent Training

Train a simple agent with calculator tool on GSM8K:

```bash
# Setup Ray cluster
ray start --head --node-ip-address="$(hostname --ip-address)" --port=6379 --num-cpus=192 --num-gpus=1

# Training command
python -m agentfly.cli train \
    algorithm.adv_estimator=grpo \
    data.train_files="./data/rlhf/math/gsm8k_train.json" \
    data.val_files="./data/rlhf/math/gsm8k_test.json" \
    data.train_batch_size=64 \
    agent.agent_type=hf \
    agent.tools="[calculator]" \
    agent.template=qwen2.5 \
    agent.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
    agent.max_turns=3 \
    agent.reward_name="math_reward_string_equal" \
    agent.num_chains=8 \
    agent.use_agent=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=256 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.model.path=Qwen/Qwen2.5-3B-Instruct \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=AgentRL \
    trainer.experiment_name=test_gsm8k \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_training_steps=100
```

### ReAct Agent with Search Tools

Train a ReAct agent with search capabilities:

```bash
python -m agentfly.cli train \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files="./data/rlhf/qa/train_random_8000.json" \
    data.val_files="./data/rlhf/qa/dev_random_500.json" \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    agent.agent_type=react \
    agent.tools="[google_search,answer]" \
    agent.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
    agent.max_turns=4 \
    agent.reward_name="qa_f1_reward" \
    agent.num_chains=1 \
    agent.use_agent=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.model.path=Qwen/Qwen2.5-3B-Instruct \
    critic.ppo_mini_batch_size=32 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=AgentRL \
    trainer.experiment_name=react_search_agent \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_training_steps=200 \
    trainer.val_before_train=True
```

### Code Agent Training

Train an agent for code generation tasks:

```bash
python -m agentfly.cli train \
    algorithm.adv_estimator=grpo \
    data.train_files="./data/rlhf/code/train.json" \
    data.val_files="./data/rlhf/code/val.json" \
    data.train_batch_size=64 \
    agent.agent_type=code \
    agent.tools="[python_executor]" \
    agent.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
    agent.max_turns=5 \
    agent.reward_name="code_reward" \
    agent.num_chains=4 \
    agent.use_agent=True \
    actor_rollout_ref.actor.optim.lr=4e-7 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.model.path=Qwen/Qwen2.5-3B-Instruct \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=AgentRL \
    trainer.experiment_name=code_agent \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_training_steps=300
```

## Training Data Format

Training data should be in JSON format with the following structure:

```json
[
    {
        "question": "Your task query here",
        "answer": "Expected answer or ground truth"
    },
    ...
]
```

The `question` field is used to form input messages to the agent, while other fields (like `answer`) are passed to the reward function for evaluation.

## Distributed Training

For multi-node training, set up Ray cluster across nodes:

```bash
# On head node
ray start --head --node-ip-address=<head_ip> --port=6379

# On worker nodes
ray start --address=<head_ip>:6379
```

Then set `trainer.nnodes` and `trainer.n_gpus_per_node` accordingly.

## Checkpointing and Resuming

- **Auto-resume**: Set `trainer.resume_mode=auto` to automatically resume from the latest checkpoint
- **Manual resume**: Set `trainer.resume_mode=resume_path` and `trainer.resume_from_path=<checkpoint_path>`
- **Checkpoint frequency**: Control with `trainer.save_freq` (saves every N iterations)

## Monitoring Training

Training metrics are logged to:
- **Console**: Always enabled
- **Weights & Biases**: Enable with `trainer.logger=['console','wandb']` and set `WANDB_API_KEY` environment variable

Key metrics tracked:
- Reward statistics (mean, std, min, max)
- Policy loss
- Value loss
- KL divergence
- Entropy
- Learning rate

## Tips and Best Practices

1. **Start Small**: Begin with small models and batch sizes to verify setup
2. **Monitor KL Divergence**: Keep KL divergence in check; adjust `kl_coef` if it grows too large
3. **Tune Learning Rate**: Start with `5e-7` to `1e-6` and adjust based on training stability
4. **Batch Size**: Balance between `train_batch_size`, `ppo_mini_batch_size`, and `ppo_micro_batch_size_per_gpu`
5. **Memory Management**: Use `gpu_memory_utilization` and FSDP offloading for large models
6. **Advantage Estimator**: `grpo` works well for most cases; `gae` for on-policy scenarios
7. **Validation**: Set `trainer.val_before_train=True` to establish baseline before training
