#!/bin/bash
# ============================================================================
# WebShop, CHAIN rollout, Qwen3.5-4B — with vs without re-tokenization.
#
#   bash train_webshop_qwen35_chain_tokenids.sh on  [gigpo|grpo] [tag]   # train on the sampled token ids
#   bash train_webshop_qwen35_chain_tokenids.sh off [gigpo|grpo] [tag]   # re-encode every turn from text
#
# Identical to `train_webshop_qwen35_chain.sh gigpo 4B` (thinking on, lr=5e-7, max_response=4096,
# max_model_len=16384, n=8, train_batch=16, 150 steps) except the ONE manipulated factor:
#   agent.run_config.use_token_ids — on: generated messages carry the sampled token ids and
#   chat-bricks splices them into both the sampling prompts and the training rows; off: no ids are
#   carried, so every turn (earlier turns in the context and the trained response) is re-encoded
#   from its text.
# The per-step token-drift line in the log covers every row (see extract_drift.py); the off arm
# carries no ids, so its drift line reports frac_rows_with_ids=0 only.
# ============================================================================

ids=${1:-on}
arm=${2:-gigpo}
tag=${3:-}                          # optional suffix, e.g. "seed2": keeps a repeat run's
                                    # wandb name and checkpoints separate from the original
size=4B
if   [[ "$ids" == "on"  ]]; then use_token_ids=true
elif [[ "$ids" == "off" ]]; then use_token_ids=false
else echo "usage: $0 [on|off] [gigpo|grpo] [tag]"; exit 1; fi
if   [[ "$arm" == "gigpo" ]]; then step_advantage_w=1.0; gigpo_mode=mean_norm
elif [[ "$arm" == "grpo"  ]]; then step_advantage_w=0.0; gigpo_mode=mean_std_norm
else echo "usage: $0 [on|off] [gigpo|grpo] [tag]"; exit 1; fi

export VLLM_USE_V1=1
# See train_webshop_gigpo.sh: vLLM holds its fraction during training, so 7B (16k context, vocab-sized
# logits) gets a smaller KV cache to leave the actor room. Rollout cache only, not the training math.
gpu_memory_utilization=0.75
[[ "$size" == "9B" ]] && gpu_memory_utilization=0.6
export WEBSHOP_IMAGE=reasonwang/webshop-env:small   # before ray start (workers inherit it)
# Tasks replayed from verl-agent's own env workers and draws (see
# examples/data_preprocess/prepare_webshop_verl_agent.py): the train file is verl-agent's exact
# training tasks in order (16 per step x 150 steps, read with data.shuffle=False); the val file is
# the 512 tasks of its first two evaluations. Each row carries its goal + product prices, served per
# session by the mounted simulator server (AF_WEBSHOP_SERVER_MOUNT).
train_dataset="./data/rlhf/webshop/webshop_train_verl_agent_small.json"
eval_dataset="./data/rlhf/webshop/webshop_val_verl_agent_small.json"
set -x

head_node_ip=$(hostname --ip-address)
port=6379
export HYDRA_FULL_ERROR=1
ray stop
rm -rf /tmp/ray/ray_current_cluster
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 192 --num-gpus 8

# --- matched to the step arm ---------------------------------------------------------
model=Qwen/Qwen3.5-${size}
lr=5e-7
gamma=0.95
penalty_coef=0.1
num_chains=8
train_batch_size=16                 # x8 chains = 128 trajectories per step
val_batch_size=512                  # verl-agent's first two evaluations (2 x 256)
ppo_mini_batch_size=8               # TRAJECTORY rows: 16 optimizer steps per training step
max_turns=15
max_response_length=4096                # thinking on (see header)
max_model_len=16384                 # Qwen3.5 thinking: rows reach ~14k prompt + 10k response tokens, but 24k OOMs
                                    # in the actor update; at 16k ~3% of chains end at max_model_len (smoke test).
kl_coef=0.01
kl_loss_type=low_var_kl

tools="[webshop_browser_action]"
reward_name="webshop_reward"  # 10 x success (score == 1.0); task_score = dense score
agent_type=action
template=null
# verl-agent's WebShop instruction as the system message; the task, the page and the
# admissible actions arrive in the user/tool turns.
system_prompt="You are an expert autonomous agent operating in the WebShop e‑commerce environment.

You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for the current step and present it within <action> </action> tags."

total_training_steps=150            # matched to the step arm (verl-agent total_epochs=150)
experiment_name="Qwen3.5-${size}-webshop-chain-${arm}-tokenids-${ids}${tag:+-$tag}"
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
    data.shuffle=False \
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
    agent.run_config.use_token_ids=$use_token_ids \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
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
