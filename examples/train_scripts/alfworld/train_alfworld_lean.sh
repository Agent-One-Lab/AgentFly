#!/bin/bash
# ============================================================================
# ALFWorld A/B: GRPO vs multi-turn GiGPO on a *lean* (verl-agent-faithful) setup.
#
# Motivation: our default ALFWorld prompt enumerates all six task categories and
# gives per-category strategy, and the agent sees full history. That makes the task
# easy enough that GRPO alone saturates (7B GRPO hit 91.8% val_unseen — ABOVE the
# GiGPO paper's GiGPO-7B of 90.8%), leaving GiGPO no headroom. To test whether our
# GiGPO implementation actually reproduces the paper's gain, we must remove the
# headroom by matching verl-agent's LEAN setup, where their GRPO-7B is only 77.6%
# and GiGPO opens a ~+13 pt gap.
#
# This ONE script drives both arms so they are byte-identical except adv_estimator:
#     bash train_alfworld_lean.sh grpo     # baseline
#     bash train_alfworld_lean.sh gigpo    # multi-turn GiGPO
#
# Faithful-to-verl-agent knobs matched here: lean generic prompt (no category hints),
# Qwen2.5-1.5B-Instruct, max_turns=50, lr=1e-6, KL 0.01 low_var_kl, group=8, and the
# -0.1 invalid-action penalty (already applied inside the alfworld_step tool's
# step_reward). Documented DELTAS we do NOT match (kept for infra simplicity; identical
# across both arms so they don't affect the internal GRPO-vs-GiGPO comparison):
#   - full conversation history (verl-agent caps at history_length=2)
#   - train_batch_size=32 / entropy_coeff=0.001 (verl-agent: 16 / unset)
# ============================================================================

# --- A/B arm: grpo (default) or gigpo (validate BEFORE touching Ray) ------------------
adv_estimator=${1:-grpo}
if [[ "$adv_estimator" != "grpo" && "$adv_estimator" != "gigpo" ]]; then
    echo "usage: $0 [grpo|gigpo]"; exit 1
fi

# Run in single node
export VLLM_USE_V1=1

set -x

head_node_ip=$(hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export HYDRA_FULL_ERROR=1
# Remove existing Ray cluster
ray stop
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 192 --num-gpus 8

# AlfWorld Configuration (verl-agent-faithful, 1.5B)
model=Qwen/Qwen2.5-3B-Instruct
# verl-agent uses 1e-6, but at 3B on our FULL-history (~11.8k ctx) lean setup that LR
# drives an entropy/KL collapse around step ~50 (entropy->0.27, KL->1.0, grad_norm->30,
# response_length 1200->400, reward crashes). 5e-7 is our proven-stable 3B value
# (rich-prompt runs climbed smoothly to 0.84 with it). Documented divergence from
# verl-agent, in service of a stable baseline; identical across both A/B arms.
lr=5e-7
val_batch_size=256
train_batch_size=32
num_chains=8                        # group size (verl-agent env.rollout.n=8)
mini_batch_size=$((train_batch_size * num_chains))

max_new_tokens_per_turn=256
max_model_len=12288
kl_coef=0.01                        # verl-agent kl_loss_coef=0.01
kl_loss_type=low_var_kl             # verl-agent kl_loss_type=low_var_kl
train_dataset="./data/rlhf/alfworld/alfworld_train_tasks.json"
eval_dataset="./data/rlhf/alfworld/alfworld_valid_unseen_tasks.json"
tools="[alfworld_step]"
reward_name="alfworld_episode_reward"
# gamma discounts the per-turn step return-to-go (GiGPO step level). GRPO ignores it,
# so this only affects the gigpo arm. verl-agent uses 0.95.
gamma=0.95

# LEAN prompt — verl-agent's generic ALFWORLD instruction. NO task-category
# enumeration, NO per-category strategy, NO step-minimization hint. The task
# description and admissible actions are delivered per-turn via the tool observation
# (alfworld_step's format_observation appends the admissible-action menu), matching
# verl-agent's per-step template content.
system_prompt="You are an expert agent operating in the ALFRED Embodied Environment.

You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for the current step and present it within <action> </action> tags."

entropy_coeff=0.001
agent_type=action
template="action-agent"
max_turns=50
total_training_steps=200
experiment_name="${model}-alfworld-lean-${max_turns}turns-${adv_estimator}-test"
project_name="Open"

python3 -m agentfly.cli train \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.gamma=$gamma \
    data.train_files=${train_dataset} \
    data.val_files=${eval_dataset} \
    data.val_batch_size=$val_batch_size \
    data.train_batch_size=$train_batch_size \
    agent.use_agent=True \
    agent.init_config.agent_type=$agent_type \
    agent.init_config.model_name_or_path=$model \
    agent.init_config.template=$template \
    "agent.init_config.system_prompt=\"${system_prompt}\"" \
    agent.init_config.max_model_len=$max_model_len \
    agent.init_config.tools=${tools} \
    agent.init_config.reward_name=${reward_name} \
    agent.run_config.generation_config.max_tokens=$max_new_tokens_per_turn \
    agent.run_config.max_turns=${max_turns} \
    agent.run_config.num_chains=$num_chains \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$model \
    critic.ppo_mini_batch_size=$mini_batch_size \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=True
