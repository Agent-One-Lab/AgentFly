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
export ENROOT_IMAGES_PATH=/mnt/weka/home/renxi.wang/Agent-One-Lab/enroot-py/data/images/r2e-gym-lite


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

# export VERL_LOGGING_LEVEL=DEBUG
# Keep the 2-node Ray cluster started above; do not run "ray stop" / "ray start --head"
# here or the cluster will shrink to one node (8 GPUs) and trainer will fail expecting 16.

model=Qwen/Qwen3-32B
lr=4e-7
max_model_len=32768
max_new_tokens_per_turn=2048
val_batch_size=512
train_batch_size=16
num_chains=4
mini_batch_size=$((train_batch_size * num_chains))
kl_coef=0.001
train_dataset="./data/rlhf/os/r2e-gym-lite.json"
eval_dataset="./data/rlhf/os/r2e-gym-lite.json"
tools="[run_shell_command]"
reward_name="r2e_gym_reward"
train_on_last_turn=False
experiment_name="test_swe_train"

# Long-context: use_remove_padding=True + ulysses_sequence_parallel_size=4 splits the sequence across
# 4 GPUs so activation memory per GPU is ~4x lower while keeping full max_model_len (e.g. 32768).
# If still OOM, try ulysses_sequence_parallel_size=8 (must divide n_gpus; 32/8=4 DP replicas).

# adv_estimator=rloo
# adv_estimator=reinforce_plus_plus
# adv_estimator=remax
adv_estimator=grpo
# adv_estimator=gae
entropy_coeff=0.001
kl_loss_type=mse
agent_type=bash_swe
max_turns=40
template="qwen3-miniswe"
tool_parser_name="hermes"
total_training_steps=100
lr_warmup_steps_ratio=0.02
project_name="Resource"

python -m agentfly.cli train \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=${train_dataset} \
    data.val_files=${eval_dataset} \
    data.val_batch_size=$val_batch_size \
    data.train_batch_size=$train_batch_size \
    agent.train_on_last_turn=$train_on_last_turn \
    agent.use_agent=True \
    agent.init_config.agent_type=$agent_type \
    agent.init_config.model_name_or_path=$model \
    agent.init_config.template=$template \
    agent.init_config.max_model_len=$max_model_len \
    agent.init_config.tool_parser_name=$tool_parser_name \
    agent.init_config.tools=${tools} \
    agent.init_config.reward_name=${reward_name} \
    agent.generation_config.max_tokens=$max_new_tokens_per_turn \
    agent.max_turns=${max_turns} \
    agent.num_chains=$num_chains \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=$model \
    critic.ppo_mini_batch_size=$train_batch_size \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.enable_activation_offload=True \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${worker_num} \
    trainer.save_freq=25 \
    trainer.test_freq=200 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False
