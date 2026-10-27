#!/bin/bash
# ALFWorld with chain rollout and GRPO: a small starting point.
#   bash examples/train_scripts/alfworld/train_alfworld.sh [extra Hydra overrides...]
# Run from the repository root. GPUs: all visible ones (restrict with CUDA_VISIBLE_DEVICES).
# The ALFWorld environments run in enroot containers (one per rollout, pooled).
# For the full step-vs-chain / GRPO-vs-GiGPO study setup, see train_alfworld_chain.sh
# and train_alfworld_gigpo.sh.
set -e
# A private Ray cluster that ends with the run (never touches other Ray clusters on the node).
export RAY_ADDRESS=${RAY_ADDRESS:-local}

model=Qwen/Qwen2.5-1.5B-Instruct
n_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
train_batch_size=16          # tasks per training step
num_chains=8                 # rollouts per task (the GRPO group)
max_turns=50
max_response_length=512      # tokens per turn
max_model_len=16384          # the whole conversation: ~50 turns of observation + response

# Downloaded from Agent-One/AgentFly-Train on first use if missing.
train_dataset="./data/rlhf/alfworld/alfworld_train_tasks.json"
eval_dataset="./data/rlhf/alfworld/alfworld_valid_seen_tasks.json"

# The ActionAgent (agent_type=action) reads the action from <action>...</action> in the
# model's reply and sends it to alfworld_step, so the prompt must ask for that format.
# The task and the admissible actions arrive in each observation.
system_prompt="You are an expert agent operating in the ALFRED Embodied Environment.

You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for the current step and present it within <action> </action> tags."

agentfly train \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_dataset} \
    data.val_files=${eval_dataset} \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=128 \
    agent.use_agent=True \
    agent.init_config.agent_type=action \
    agent.init_config.model_name_or_path=$model \
    agent.init_config.template=null \
    "agent.init_config.system_prompt=\"${system_prompt}\"" \
    agent.init_config.max_model_len=$max_model_len \
    agent.init_config.tools="[alfworld_step]" \
    agent.init_config.reward_name=alfworld_reward \
    agent.run_config.rollout=chain \
    "agent.run_config.rollout_config={project_actions: true}" \
    agent.run_config.generation_config.max_tokens=$max_response_length \
    agent.run_config.generation_config.temperature=1.0 \
    agent.run_config.max_turns=$max_turns \
    agent.run_config.num_chains=$num_chains \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((train_batch_size * num_chains)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    critic.model.path=$model \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=AgentFly \
    trainer.experiment_name=alfworld_chain_grpo \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=200 \
    trainer.val_before_train=True \
    "$@"
