#!/bin/bash
# ============================================================================
# ALFWorld — CHAIN rollout arm (full conversation history) for the
# step-vs-chain / GRPO-vs-GiGPO / model-size study.
#
#   bash train_alfworld_chain.sh grpo  [1.5B|3B|7B]   # multi-turn GRPO
#   bash train_alfworld_chain.sh gigpo [1.5B|3B|7B]   # multi-turn GiGPO (step advantage
#                                                     #   scattered onto turn spans)
#
# Everything the OPTIMIZER and ENVIRONMENT see is matched to train_alfworld_gigpo.sh
# (the step arm): model, lr=1e-6, KL 0.01 low_var_kl, entropy 0.001, clip 0.2, gamma=0.95,
# gigpo_mode=mean_std_norm, penalty 0.1, n=8, train_batch=16 (=128 trajectories/step),
# max_turns=50, max_response=512, train temp=1.0, val temp=0.4 sampled on valid_seen,
# always-step projection, native Qwen template, both training/inference-consistency fixes.
#
# The ONE manipulated factor is the ROLLOUT:
#   step arm : one training row per env step; the model sees a flat prompt with the last
#              history_length=2 (obs, action) pairs rendered as text (verl-agent's setup).
#   chain arm: one training row per trajectory; the model sees the FULL chat history
#              (system + every obs/response turn) and is trained on every assistant turn.
#
# DELTAS THAT FOLLOW FROM THE MANIPULATION (inherent, documented; report them, don't hide):
#   - Prompt placement: the verl-agent instruction is the SYSTEM message here (the flat
#     arm re-states it inside each user turn with Qwen's default system).
#   - Context: max_model_len=12288 (full history) vs 4096. More tokens per step -> log
#     tokens/step and wall-clock per arm (the throughput story).
#   - PPO mini-batch is in TRAJECTORY rows: ppo_mini_batch_size=8 -> 128/8 = 16 optimizer
#     steps per training step (~35k trained tokens each) vs the step arm's 24 (~23k). This
#     is the closest match the 8-GPU DP layout allows (mini-batch must be a multiple of 8
#     rows; 8 is the finest). Keep it identical across sizes and arms.
# ============================================================================

arm=${1:-grpo}
size=${2:-1.5B}
if   [[ "$arm" == "gigpo" ]]; then step_advantage_w=1.0
elif [[ "$arm" == "grpo"  ]]; then step_advantage_w=0.0
else echo "usage: $0 [grpo|gigpo] [1.5B|3B|7B]"; exit 1; fi
if [[ "$size" != "1.5B" && "$size" != "3B" && "$size" != "7B" ]]; then
    echo "usage: $0 [grpo|gigpo] [1.5B|3B|7B]"; exit 1; fi

export VLLM_USE_V1=1
set -x

head_node_ip=$(hostname --ip-address)
port=6379
export HYDRA_FULL_ERROR=1
ray stop
rm -rf /tmp/ray/ray_current_cluster
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 192 --num-gpus 8

# --- matched to the step arm ---------------------------------------------------------
model=Qwen/Qwen2.5-${size}-Instruct
lr=5e-7
gamma=0.95
gigpo_mode=mean_std_norm
penalty_coef=0.1
num_chains=8                        # group size
train_batch_size=16                 # x8 chains = 128 trajectories per step
val_batch_size=128
ppo_mini_batch_size=8               # TRAJECTORY rows (see header): 16 updates / step
max_turns=50
max_response_length=512
max_model_len=16384                 # full history: ~50 turns x (obs + response)
kl_coef=0.01
kl_loss_type=low_var_kl

train_dataset="./data/rlhf/alfworld/alfworld_train_tasks.json"
eval_dataset="./data/rlhf/alfworld/alfworld_valid_seen_tasks.json"

tools="[alfworld_step]"
reward_name="alfworld_episode_reward"   # 10 * won
agent_type=action
# Native Qwen chat template (same as the step arm). The ActionAgent renders NO tools
# block (agent.prompt_tools), and its history keeps the FULL generated text so what the
# model sees next turn == what it sampled == what training splices.
template=null
# verl-agent's generic ALFWorld instruction (no category hints), as the system message.
# The task description and admissible actions arrive per turn in the tool observation.
system_prompt="You are an expert agent operating in the ALFRED Embodied Environment.

You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for the current step and present it within <action> </action> tags."

total_training_steps=200
experiment_name="Qwen2.5-${size}-alfworld-chain-${arm}-v2"
project_name="Open"

python3 -m agentfly.cli train \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=$gamma \
    algorithm.step_advantage_w=$step_advantage_w \
    algorithm.gigpo_mode=$gigpo_mode \
    algorithm.invalid_action_penalty_coef=$penalty_coef \
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
    agent.run_config.rollout=chain \
    "agent.run_config.rollout_config={project_actions: true}" \
    agent.run_config.generation_config.max_tokens=$max_response_length \
    agent.run_config.generation_config.temperature=1.0 \
    agent.run_config.max_turns=${max_turns} \
    agent.run_config.num_chains=$num_chains \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$model \
    critic.ppo_mini_batch_size=$ppo_mini_batch_size \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=300 \
    trainer.test_freq=10 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=True
