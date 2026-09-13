#!/bin/bash
# ============================================================================
# WebShop (verl-agent-aligned, small catalog) — CHAIN rollout arm (full conversation history).
#
#   bash train_webshop_chain.sh grpo  [1.5B|3B|7B]   # multi-turn GRPO
#   bash train_webshop_chain.sh gigpo [1.5B|3B|7B]   # multi-turn GiGPO (step advantage on turn spans)
#
# Everything the ENVIRONMENT and estimator see is matched to train_webshop_gigpo.sh (the step
# arm): image, goals, val ids 0-499, reward (verl-agent's: 10 x success), penalty 0.1,
# gamma 0.95, gigpo_mode=mean_norm, n=8, train_batch=16, 150 steps, max_turns=15, max_response=512,
# train temp 1.0 / val temp 0.4, always-step projection, native Qwen template.
#
# The ONE manipulated factor is the ROLLOUT: one training row per trajectory, the model sees
# the FULL chat history (system + every page/response turn) and is trained on every
# assistant turn. Deltas that follow (report them): lr=5e-7 (the ALFWorld chain arms' value;
# the step arm uses verl-agent's 1e-6), the verl-agent instruction as the SYSTEM message,
# max_model_len sized from measured 15-turn episode lengths, PPO mini-batch of 8 trajectories.
# ============================================================================

arm=${1:-grpo}
size=${2:-1.5B}
if   [[ "$arm" == "gigpo" ]]; then step_advantage_w=1.0
elif [[ "$arm" == "grpo"  ]]; then step_advantage_w=0.0
else echo "usage: $0 [grpo|gigpo] [1.5B|3B|7B]"; exit 1; fi
if [[ "$size" != "1.5B" && "$size" != "3B" && "$size" != "7B" ]]; then
    echo "usage: $0 [grpo|gigpo] [1.5B|3B|7B]"; exit 1; fi

export VLLM_USE_V1=1
export WEBSHOP_IMAGE=reasonwang/webshop-env:small   # before ray start (workers inherit it)
train_dataset="./data/rlhf/webshop/webshop_train_small.json"
eval_dataset="./data/rlhf/webshop/webshop_val_small.json"
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
gigpo_mode=mean_norm
penalty_coef=0.1
num_chains=8
train_batch_size=16                 # x8 chains = 128 trajectories per step
val_batch_size=500                  # all val task ids 0-499
ppo_mini_batch_size=8               # TRAJECTORY rows: 16 optimizer steps per training step
max_turns=15
max_response_length=512
max_model_len=16384                 # measured on 80 chain-rollout episodes from the July small-image
                                    # runs (3B/7B): full conversations p50 2.6-2.8k, p99 7.8k, max 7.8k
                                    # tokens; 15-turn episodes max ~3k. 16k = >2x the observed max and
                                    # covers 15 x 512-token rambling turns + pages; same as the ALFWorld chain arms.
kl_coef=0.01
kl_loss_type=low_var_kl

tools="[webshop_browser_action]"
reward_name="webshop_episode_reward"  # 10 x success (score == 1.0); task_score = dense score
agent_type=action
template=null
# verl-agent's WebShop instruction as the system message; the task, the page and the
# admissible actions arrive in the user/tool turns.
system_prompt="You are an expert autonomous agent operating in the WebShop e‑commerce environment.

You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for the current step and present it within <action> </action> tags."

total_training_steps=150            # matched to the step arm (verl-agent total_epochs=150)
experiment_name="Qwen2.5-${size}-webshop-chain-${arm}-v2"
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
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=True
