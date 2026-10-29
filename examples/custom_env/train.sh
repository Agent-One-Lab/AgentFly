#!/bin/bash
# Train an agent on the number-guessing environment (examples/custom_env).
#   python examples/custom_env/make_data.py
#   bash examples/custom_env/train.sh [extra Hydra overrides...]
# GPUs: all visible ones (restrict with CUDA_VISIBLE_DEVICES).
set -e
# The tool, reward and env are loaded by import path ("my_task.tools:guess",
# "my_task.env.GuessNumberEnv") in the Ray workers, which start in the launch folder:
# run from the folder that contains my_task.
cd "$(dirname "${BASH_SOURCE[0]}")"
# A private Ray cluster that ends with the run (never touches other Ray clusters on the node).
export RAY_ADDRESS=${RAY_ADDRESS:-local}

model=Qwen/Qwen2.5-3B-Instruct
n_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
train_batch_size=32
group_size=8

agentfly train \
    algorithm.adv_estimator=grpo \
    data.train_files="./data/guess_train.json" \
    data.val_files="./data/guess_val.json" \
    data.train_batch_size=$train_batch_size \
    agent.use_agent=True \
    agent.init_config.agent_type=hf \
    agent.init_config.model_name_or_path=$model \
    agent.init_config.tools="[my_task.tools:guess]" \
    agent.init_config.reward_name=my_task.tools:guess_reward \
    agent.init_config.max_model_len=4096 \
    agent.run_config.max_turns=10 \
    agent.run_config.group_size=$group_size \
    agent.run_config.generation_config.max_tokens=256 \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((train_batch_size * group_size)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.model.path=$model \
    trainer.logger="['console','wandb']" \
    trainer.project_name=AgentFly \
    trainer.experiment_name=guess_number \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=50 \
    "$@"
