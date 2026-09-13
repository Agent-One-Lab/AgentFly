#!/bin/bash
# ============================================================================
# ALFWorld GiGPO — faithful reproduction of verl-agent's run_alfworld.sh.
#
# Uses the verl-agent-style rollout: StepRollout (one training row per env step) +
# FlatAlfworldPromptBuilder (single flat prompt from ALFWORLD_TEMPLATE, byte-identical
# to verl-agent) + the per-step GiGPO estimator (core_gigpo_step, numerically matched
# to verl-agent's gigpo/core_gigpo.py). Reward = 10 * won (verl-agent's compute_reward).
#
#   bash train_alfworld_gigpo.sh gigpo [1.5B|3B|7B]   # full GiGPO (step_advantage_w=1.0)
#   bash train_alfworld_gigpo.sh grpo  [1.5B|3B|7B]   # baseline: step_advantage_w=0 (pure GRPO)
#
# Matched to verl-agent run_alfworld.sh: Qwen2.5-{size}, lr=1e-6, gamma=0.95,
# step_advantage_w=1.0, gigpo_mode=mean_std_norm, invalid_action_penalty_coef=0.1,
# rollout n=8, history_length=2, max_steps=50, KL 0.01 low_var_kl, max_prompt=2048 /
# max_response=512, train temp=1.0, eval=valid_seen (in-distribution).
#
# MATCHED to verl-agent:
#   - Always-step projection (rollout_config.project_actions=true): every turn steps the env
#     (a turn with no extractable <action> falls back to last-30-chars garbage and steps
#     anyway); the episode ends ONLY on env done (won/lost) or max_turns — never on a
#     malformed / no-tool-call turn. is_action_valid is verl-agent's FORMAT check
#     (<action> + <think> + no Chinese), not the env-semantic "nothing happens" flag.
#
# ALSO MATCHED (verified on run jubs5csy vs verl-agent u4i879dl):
#   - Training tokens: the sampled ids are spliced (message token_ids), and the prompt is
#     rendered identically for sampling and training (agent.prompt_tools; ActionAgent
#     renders no tools block). See docs/features/training_consistency.md.
#   - Validation: temp 0.4, sampled (rollout.val_kwargs; the trainer honors it), like
#     verl-agent's reported numbers. test_freq=10 (theirs 5).
#   - PPO mini-batch: 256 step-rows -> ~24 optimizer steps per training step, same as
#     verl-agent (their ppo_mini_batch_size is normalized the same way).
# ============================================================================

# --- arm: gigpo (default) or grpo (step_advantage_w=0); size: 1.5B (default) | 3B | 7B --
arm=${1:-gigpo}
size=${2:-1.5B}
if   [[ "$arm" == "gigpo" ]]; then step_advantage_w=1.0
elif [[ "$arm" == "grpo"  ]]; then step_advantage_w=0.0
else echo "usage: $0 [gigpo|grpo] [1.5B|3B|7B]"; exit 1; fi
if [[ "$size" != "1.5B" && "$size" != "3B" && "$size" != "7B" ]]; then
    echo "usage: $0 [gigpo|grpo] [1.5B|3B|7B]"; exit 1; fi

export VLLM_USE_V1=1
set -x

head_node_ip=$(hostname --ip-address)
port=6379
export HYDRA_FULL_ERROR=1
ray stop
rm -rf /tmp/ray/ray_current_cluster
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 192 --num-gpus 8

# --- verl-agent-faithful config (run_alfworld.sh) ------------------------------------
model=Qwen/Qwen2.5-${size}-Instruct
lr=1e-6
gamma=0.95                          # GiGPO step return-to-go discount
gigpo_mode=mean_std_norm
penalty_coef=0.1                    # invalid-action penalty (post-discount)
num_chains=8                        # rollout n (group size)
train_batch_size=16
val_batch_size=128
ppo_mini_batch_size=256             # verl-agent ppo_mini_batch_size
max_turns=50                        # verl-agent env.max_steps
history_length=2                    # verl-agent env.history_length
max_prompt_length=2048
max_response_length=512
max_model_len=4096                  # prompt(<=2048) + response(512) + headroom
kl_coef=0.01
kl_loss_type=low_var_kl

train_dataset="./data/rlhf/alfworld/alfworld_train_tasks.json"
# verl-agent evaluates in-distribution (valid_seen). Build it with:
#   python examples/data_preprocess/prepare_alfworld.py --splits valid_seen
eval_dataset="./data/rlhf/alfworld/alfworld_valid_seen_tasks.json"

tools="[alfworld_step]"
reward_name="alfworld_episode_reward"   # now 10 * won (verl-agent scale)
agent_type=action
# Use the model's NATIVE chat template: leaving this unset (null) makes chat-bricks fall
# back to the model name -> Qwen2.5's stock HF template, which auto-renders Qwen's default
# system ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") — matching
# verl-agent exactly (they use the stock tokenizer template). The custom "action-agent"
# template would instead emit an EMPTY system block, leaving the instruction-tuned base
# off-distribution. The action PARSER (<think>/<action> regex) is independent of the template,
# and the flat builder's per-step assistant tokens are unchanged (verified).
template=null
# Do NOT set a system prompt (null = unset). The agent only injects a system message when
# system_prompt is truthy, so an empty string would be inert too — but null makes the intent
# explicit: nothing is injected, so the model's NATIVE template renders its own default system
# ("You are Qwen, ..."). Injecting an empty string instead would emit an EMPTY <|im_start|>system
# block and OVERRIDE that default. verl-agent likewise sets no system prompt.
system_prompt=null

total_training_steps=200
# step = StepRollout + flat verl-agent prompt (history_length=2); v2 = both consistency fixes in
experiment_name="Qwen2.5-${size}-alfworld-step-${arm}-v2"
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
    "agent.run_config.rollout_config={prompt_builder: alfworld_flat, history_length: ${history_length}, max_prompt_length: ${max_prompt_length}, project_actions: true}" \
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
