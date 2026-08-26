#!/bin/bash
#SBATCH --job-name=alfworld_gigpo
#SBATCH --time=200:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --account=iq
#SBATCH --cpus-per-task=128
#SBATCH --output=stdout/%x_%j.out
#SBATCH --error=stdout/%x_%j.err

# Multi-turn GiGPO on ALFWorld, 2 nodes (16 GPUs). A/B baseline is adv_estimator=grpo.
# Get the list of allocated nodes
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "Nodes: ${nodes[@]}"

set -x

export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export worker_num=$SLURM_NNODES
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

# =================== Ray start ===================
# ray stop on all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ray stop
sleep 10
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
    uv run --no-sync ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &
sleep 10

# Worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
        uv run --no-sync ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &
done
sleep 10

# =================== AlfWorld / GiGPO config ===================
model=Qwen/Qwen2.5-3B-Instruct
lr=5e-7
val_batch_size=256
train_batch_size=8
num_chains=8
mini_batch_size=$((train_batch_size * num_chains))   # 64

max_new_tokens_per_turn=256
max_model_len=40000
kl_coef=0.001
train_dataset="./data/rlhf/alfworld/alfworld_train_tasks.json"
eval_dataset="./data/rlhf/alfworld/alfworld_valid_unseen_tasks.json"
tools="[alfworld_step]"
reward_name="alfworld_episode_reward"
adv_estimator=gigpo
gamma=0.95                          # GiGPO step-return discount (GRPO ignores gamma)

# --- Sequence parallelism (optional; likely UNNEEDED for 3B @ 40k with micro_batch=1) ---
# If you OOM (or want micro_batch>1), enable Ulysses SP by adding these to the launch:
#   actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
#   actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
#   critic.ulysses_sequence_parallel_size=4 \
# SP splits the 40k sequence across GPUs. Note: the long-context SP path typically also
# wants actor_rollout_ref.model.use_remove_padding=True (currently False for the action
# agent) — flip it together and verify the multi-turn masking still lines up.

system_prompt="You are an ALFWorld agent operating in an interactive, text-based household environment (derived from the ALFRED benchmark). You are placed in one of 120 possible rooms (kitchen, bedroom, bathroom, or living room) populated with portable objects (e.g., apple, mug, book) and static receptacles (e.g., microwave, fridge, drawer, countertop). Your goal is to complete the given household task by interacting with the world through high-level text commands, and to finish it in as few steps as possible. The environment is partially observable: the initial observation lists all navigable receptacles in the room, but you must actively go to receptacles, open them, and examine their contents to find target objects.

You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, you can do one action by <action> action </action>. If you think you have finished the task, summarize what you have done.

ALFWorld tasks fall into six categories. Recognize your task type from the goal and plan accordingly:
- Pick & Place (e.g., put a plate on the coffee table): find the object, pick it up, go to the target receptacle, put it down.
- Examine in Light (e.g., examine a book under the lamp): find the object, pick it up, go to a lamp, turn on the lamp while holding the object.
- Clean & Place (e.g., put a clean knife in the drawer): find the object, pick it up, go to a sinkbasin, clean it there, then place it at the target receptacle.
- Heat & Place (e.g., put a hot mug on the coffee table): find the object, pick it up, go to a microwave, heat it there, then place it at the target receptacle.
- Cool & Place (e.g., put a cool bottle on the countertop): find the object, pick it up, go to a fridge, cool it there, then place it at the target receptacle.
- Pick Two & Place (e.g., put two pencils in the drawer): find one instance, place it at the target; then find a second instance, place it there too.

Each observation lists the valid actions for the current state under 'Admissible actions:'. Always choose your next action from that list — any command not on it will fail with 'Nothing happens'. Objects and receptacles are referred to by class name plus an ID (e.g., mug 1, countertop 2).

Remember that you must put your action inside <action> and </action> tags."

entropy_coeff=0.001
kl_loss_type=mse
agent_type=action
template="action-agent"
max_turns=40
total_training_steps=200
experiment_name="${model}-alfworld-${max_turns}turns-${adv_estimator}-2node-40k"
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
    trainer.nnodes=${worker_num} \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=True
