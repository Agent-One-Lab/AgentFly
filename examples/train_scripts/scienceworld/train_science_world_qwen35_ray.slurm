#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=200:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --account=iq
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --output=stdout/%x_%j.out
#SBATCH --error=stdout/%x_%j.err

# Get the list of allocated nodes
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "Nodes to check: ${nodes[@]}"

set -x

# We'll track PIDs so we can wait on them and detect errors
declare -A pids
export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export worker_num=$SLURM_NNODES
export TRITON_HOME=/tmp/triton_cache
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1


# =================== Ray start ===================
# ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
        ${CONDA_BIN_PATH}ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &
done
sleep 10


system_prompt="You are a ScienceWorld agent operating in an interactive, text-based environment that simulates elementary-school science tasks (e.g., thermodynamics, simple circuits, chemistry, biology). Your goal is to complete the current task by interacting with the world through text commands, earning the highest possible task score, and finishing efficiently. The environment is partially observable; you must actively examine rooms, containers, and your inventory to gather needed information.
You must conduct reasoning inside <think> and </think> first every time you get new information.

AVAILABLE ACTIONS (you may use these; some take 0, 1, or 2 arguments):
Core navigation & sensing:
- go to [location]: move to a new location
- look around: describe the current room
- look at [object]: describe an object in detail
- look in [container]: describe a container's contents
- read [object]: read a note or book
- focus on [object]: signal intent on a task object
- task: describe current task
- inventory: list agent's inventory
- wait [duration]: take no action for some duration

Object manipulation:
- pick up [object]: move an object to the inventory
- put down [object]: drop an inventory item
- move [object] to [container]: move an object to a container
- open [object]: open a container
- close [object]: close a container
- activate [object]: activate a device
- deactivate [object]: deactivate a device
- use [tool] [on [object]]: use a device/item

Liquids & chemistry:
- pour [liquid/container] into [container]: pour a liquid into a container
- dunk [container] into [liquid]: dunk a container into a liquid
- mix [container]: chemically mix a container

Living things / misc:
- eat [object]: eat a food
- flush [object]: flush a toilet

Electricity (for simple circuits):
- connect [object] to [object]: connect electrical components
- disconnect [object]: disconnect electrical components"

model=Qwen/Qwen3.5-4B
template=qwen3-think
tool_parser_name="qwen3_coder"

lr=4e-7
max_model_len=16384
max_new_tokens_per_turn=512
val_batch_size=512
batch_size=32
num_chains=4
# full on-policy
mini_batch_size=$((batch_size * num_chains))
kl_coef=0.001
train_dataset="./data/rlhf/scienceworld/scienceworld_train.json"
eval_dataset="./data/rlhf/scienceworld/scienceworld_test.json"
# adv_estimator=rloo
# adv_estimator=reinforce_plus_plus
# adv_estimator=remax
# adv_estimator=grpo
adv_estimator=gae
# adv_estimator=contextrl
use_critic=True

agent_type=hf
tools="[scienceworld_explorer]"
reward_name="scienceworld_reward"

entropy_coeff=0.001
kl_loss_type=mse
max_turns=30
lr_warmup_steps_ratio=0.08

use_remove_padding=True
sequence_parallel_size=4

critic_lr=1e-5
gamma=0.99
lam=0.95

total_training_steps=300
model_base_name=$(basename $model)
project_name="Context"
experiment_name="scienceworld_${model_base_name}_${adv_estimator}_new"

python -m agentfly.cli train \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=${train_dataset} \
    data.val_files=${eval_dataset} \
    data.val_batch_size=$val_batch_size \
    data.train_batch_size=$batch_size \
    agent.use_agent=True \
    agent.init_config.agent_type=$agent_type \
    "agent.init_config.system_prompt=\"${system_prompt}\"" \
    agent.init_config.tool_parser_name=${tool_parser_name} \
    agent.init_config.max_model_len=$max_model_len \
    agent.init_config.tools=$tools \
    agent.init_config.template=$template \
    agent.init_config.model_name_or_path=$model \
    agent.init_config.reward_name=$reward_name \
    agent.run_config.generation_config.max_tokens=$max_new_tokens_per_turn \
    agent.run_config.max_turns=${max_turns} \
    agent.run_config.num_chains=$num_chains \
    agent.run_config.context_config.resource_backend=ray \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$sequence_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$sequence_parallel_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    critic.enable=$use_critic \
    critic.model.path=$model \
    critic.fsdp.model_dtype=bf16 \
    critic.optim.lr=$critic_lr \
    critic.ppo_mini_batch_size=32 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gamma \
    algorithm.lam=$lam \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${worker_num} \
    trainer.save_freq=50 \
    trainer.test_freq=300 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False
