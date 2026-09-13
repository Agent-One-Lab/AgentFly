#!/bin/bash
# ============================================================================
# WebShop (verl-agent-aligned, small catalog) — STEP rollout arm.
#
#   bash train_webshop_gigpo.sh gigpo [1.5B|3B|7B]   # GiGPO (step_advantage_w=1.0)
#   bash train_webshop_gigpo.sh grpo  [1.5B|3B|7B]   # baseline: step_advantage_w=0 (pure GRPO)
#
# Matched to verl-agent examples/gigpo_trainer/run_webshop.sh: Qwen2.5-{size}-Instruct,
# lr=1e-6, gamma=0.95, gigpo_mode=mean_norm (their WebShop choice), invalid-action penalty 0.1,
# rollout n=8, history_length=2, max_steps=15, KL 0.01 low_var_kl, max_prompt=4096 /
# max_response=512, ppo_mini_batch_size=64 step rows, 150 steps, train temp=1.0, val temp=0.4 sampled.
#
# Environment: reasonwang/webshop-env:small — 1,000-product catalog with a matching index and
# the same 6,910 synthetic goals as verl-agent (task ids byte-identical). val = ids 0-499 (all
# 500; verl-agent samples 256 of them), train = 500-6909. WEBSHOP_IMAGE must be exported
# BEFORE `ray start` so the Ray workers inherit it.
#
# Reward = verl-agent's (upstream envs.WebshopWorker.step): 10 if the purchase's score == 1.0,
# else 0 — success only, the dense score is NOT a training signal. Success is logged as
# trajectory/accuracy (val-aux/.../rm_trajectory/accuracy/mean@1 at validation; val-core reward
# = 10 x success); the dense score as task_score (val-aux/.../rm_task_score/mean@1).
#
# Rollout: StepRollout (one training row per env step) + FlatWebshopPromptBuilder (verl-agent's
# WEBSHOP_TEMPLATE, byte-identical) + always-step projection; is_action_valid is the FORMAT
# check (<think> + <action>). Sampling prompts are rendered with the training tokenizer and
# executed as ids by verl (prompt_check must report identical rows).
# ============================================================================

arm=${1:-gigpo}
size=${2:-1.5B}
if   [[ "$arm" == "gigpo" ]]; then step_advantage_w=1.0
elif [[ "$arm" == "grpo"  ]]; then step_advantage_w=0.0
else echo "usage: $0 [gigpo|grpo] [1.5B|3B|7B]"; exit 1; fi
if [[ "$size" != "1.5B" && "$size" != "3B" && "$size" != "7B" ]]; then
    echo "usage: $0 [gigpo|grpo] [1.5B|3B|7B]"; exit 1; fi

export VLLM_USE_V1=1
# Select the verl-agent-aligned WebShop image + goal files (exported before ray start).
export WEBSHOP_IMAGE=reasonwang/webshop-env:small
train_dataset="./data/rlhf/webshop/webshop_train_small.json"
eval_dataset="./data/rlhf/webshop/webshop_val_small.json"
set -x

head_node_ip=$(hostname --ip-address)
port=6379
export HYDRA_FULL_ERROR=1
ray stop
rm -rf /tmp/ray/ray_current_cluster
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 192 --num-gpus 8

# --- verl-agent-faithful config (run_webshop.sh) ------------------------------------
model=Qwen/Qwen2.5-${size}-Instruct
lr=1e-6
gamma=0.95
gigpo_mode=mean_norm
penalty_coef=0.1
num_chains=8                        # rollout n (group size)
train_batch_size=16
val_batch_size=500                  # all val task ids 0-499
ppo_mini_batch_size=64              # step rows per optimizer step (verl-agent run_webshop.sh; ALFWorld uses 256)
max_turns=15                        # verl-agent env.max_steps
history_length=2                    # verl-agent env.history_length
max_prompt_length=4096
max_response_length=512
max_model_len=12288                 # typical prompt ~1.1k (per-step max 2.3-3.6k over steps 1-29); a rare
                                    # 6,961-token step prompt crashed vLLM at 4608. Prompts <=4096 are unchanged vs
                                    # verl-agent (which errors on longer ones); this only admits the rare outlier.
kl_coef=0.01
kl_loss_type=low_var_kl

tools="[webshop_browser_action]"
reward_name="webshop_episode_reward"  # 10 x success (score == 1.0); task_score = dense score
agent_type=action
template=null                       # native Qwen chat template (stock system message)
system_prompt=null                  # the flat prompt carries the instruction

total_training_steps=150            # verl-agent total_epochs=150, one batch per epoch
experiment_name="Qwen2.5-${size}-webshop-step-${arm}-v2"
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
    agent.init_config.system_prompt=$system_prompt \
    agent.init_config.max_model_len=$max_model_len \
    agent.init_config.tools=${tools} \
    agent.init_config.reward_name=${reward_name} \
    agent.run_config.rollout=step \
    "agent.run_config.rollout_config={prompt_builder: webshop_flat, history_length: ${history_length}, max_prompt_length: ${max_prompt_length}, project_actions: true}" \
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
