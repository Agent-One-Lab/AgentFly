#!/bin/bash
# ============================================================================
# WebShop: validate a trained (or base) model on a task file — no training.
#
#   bash eval_webshop.sh <model_path> <step|chain> <task_file> <experiment_name>
#
# Runs ONE validation pass (trainer.val_only) with exactly the rollout/env/reward settings of
# the matching training script (train_webshop_gigpo.sh for step, train_webshop_chain.sh for
# chain), so the number is comparable to that run's own validation curve: same image, same
# per-session goals, val temperature 0.4, one sample per task.
#
# Checkpoints are FSDP shards; merge one to HF first:
#   python -m agentfly.verl.model_merger merge --backend fsdp \
#       --local_dir checkpoints/Open/<run>/global_step_150/actor --target_dir merged_models/<run>
#
# Used for the held-out generalization set (webshop_heldout_verl_agent_small.json: goals whose
# product never appears in training), which verl-agent's own split cannot measure.
# ============================================================================

model=${1:?usage: eval_webshop.sh <model_path> <step|chain> <task_file> <experiment_name>}
rollout=${2:?usage: eval_webshop.sh <model_path> <step|chain> <task_file> <experiment_name>}
task_file=${3:?usage: eval_webshop.sh <model_path> <step|chain> <task_file> <experiment_name>}
experiment_name=${4:?usage: eval_webshop.sh <model_path> <step|chain> <task_file> <experiment_name>}

export VLLM_USE_V1=1
export WEBSHOP_IMAGE=reasonwang/webshop-env:small   # before ray start (workers inherit it)
set -x

head_node_ip=$(hostname --ip-address)
port=6379
export HYDRA_FULL_ERROR=1
ray stop
rm -rf /tmp/ray/ray_current_cluster
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 192 --num-gpus 8

val_batch_size=512
max_turns=15
max_response_length=512
tools="[webshop_browser_action]"
reward_name="webshop_reward"
project_name="Open"

# The two arms differ exactly as their training scripts do: the step arm renders verl-agent's flat
# prompt (2 turns of history) per step, the chain arm keeps the whole conversation under a system prompt.
if [[ "$rollout" == "step" ]]; then
    max_model_len=12288
    system_prompt=null
    rollout_config="{prompt_builder: webshop_flat, history_length: 2, max_prompt_length: 4096, project_actions: true}"
elif [[ "$rollout" == "chain" ]]; then
    max_model_len=16384
    system_prompt="You are an expert autonomous agent operating in the WebShop e‑commerce environment.

You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for the current step and present it within <action> </action> tags."
    rollout_config="{project_actions: true}"
else
    echo "usage: $0 <model_path> <step|chain> <task_file> <experiment_name>"; exit 1
fi

python3 -m agentfly.cli train \
    algorithm.adv_estimator=gigpo \
    data.train_files=${task_file} \
    data.val_files=${task_file} \
    data.val_batch_size=$val_batch_size \
    data.train_batch_size=16 \
    data.shuffle=False \
    agent.use_agent=True \
    agent.init_config.agent_type=action \
    agent.init_config.model_name_or_path=$model \
    agent.init_config.template=null \
    "agent.init_config.system_prompt=\"${system_prompt}\"" \
    agent.init_config.max_model_len=$max_model_len \
    agent.init_config.tools=${tools} \
    agent.init_config.reward_name=${reward_name} \
    agent.run_config.rollout=$rollout \
    "agent.run_config.rollout_config=${rollout_config}" \
    agent.run_config.generation_config.max_tokens=$max_response_length \
    agent.run_config.generation_config.temperature=1.0 \
    agent.run_config.max_turns=${max_turns} \
    agent.run_config.group_size=8 \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
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
    critic.ppo_micro_batch_size_per_gpu=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=1 \
    trainer.val_before_train=True \
    trainer.val_only=True
