#!/bin/bash
# Chess Puzzle Solving Agent Training Script
#
# This script trains an agent to solve chess puzzles using:
# - Tools: chess_move, chess_get_state, chess_get_legal_moves
# - Reward: chess_puzzle_reward (dense) or chess_puzzle_reward_simple (binary)
# - Environment: ChessPuzzleEnv (python-chess + Stockfish)
#
# Prerequisites:
# 1. Install Stockfish: brew install stockfish (macOS) or apt install stockfish (Linux)
# 2. Prepare training data in data/chess/ directory
# 3. Set WANDB_API_KEY for logging

set -x

# ============================================================================
# Environment Setup
# ============================================================================

export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_key}"
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

# Ray cluster setup
head_node_ip=$(hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

# Clean up existing Ray cluster
ray stop
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node (adjust --num-cpus and --num-gpus for your hardware)
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 32 --num-gpus 1

# ============================================================================
# Model Configuration
# ============================================================================

# Base model to fine-tune
model="Qwen/Qwen2.5-3B-Instruct"
template="qwen2.5"

# Alternative models (uncomment to use):
# model="Qwen/Qwen2.5-7B-Instruct"
# model="meta-llama/Llama-3.1-8B-Instruct"
# template="llama3"

# ============================================================================
# Agent Configuration
# ============================================================================

agent_type="react"  # ReAct agent for tool-using tasks
agent_backend="async_verl"

# Chess-specific tools
tools="[chess_move,chess_get_state,chess_get_legal_moves]"

# Reward function:
# - chess_puzzle_reward: Dense reward based on Stockfish evaluation (recommended)
# - chess_puzzle_reward_simple: Binary reward (solved/not solved)
reward_name="chess_puzzle_reward"

# Maximum turns per puzzle (moves + state checks)
max_turns=10

# Parallel rollouts per puzzle sample
num_chains=8

# ============================================================================
# Training Data
# ============================================================================

train_dataset="./data/chess/chess_puzzles_train.json"
val_dataset="./data/chess/chess_puzzles_val.json"

# ============================================================================
# Training Hyperparameters
# ============================================================================

batch_size=64
mini_batch_size=$batch_size
lr=4e-7
kl_coef=0.001
entropy_coeff=0.001
kl_loss_type="mse"
response_length=256

# Advantage estimator options: grpo, reinforce_plus_plus, rloo, remax, gae
adv_estimator="grpo"

# Training duration
total_training_steps=200
save_freq=50
test_freq=10

# ============================================================================
# Logging
# ============================================================================

project_name="AgentRL"
experiment_name="chess_puzzle_solver_$(date +%Y%m%d_%H%M%S)"

# ============================================================================
# Launch Training
# ============================================================================

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=$train_dataset \
    data.val_files=$val_dataset \
    data.train_batch_size=$batch_size \
    agent.agent_type=$agent_type \
    agent.tools=$tools \
    agent.template=$template \
    agent.model_name_or_path=$model \
    agent.max_turns=${max_turns} \
    agent.backend=${agent_backend} \
    agent.reward_name=$reward_name \
    agent.num_chains=$num_chains \
    agent.use_agent=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=$response_length \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=$model \
    critic.ppo_mini_batch_size=${mini_batch_size} \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False
