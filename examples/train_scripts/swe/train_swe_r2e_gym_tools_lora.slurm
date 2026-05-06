#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=200:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --account=iq
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
# Directory holding the r2e-gym-lite enroot images. See docs/examples/swe.md.
export ENROOT_IMAGES_PATH=/mnt/weka/home/renxi.wang/Agent-One-Lab/enroot-py/data/images/r2e-gym-lite
export SSPO_MULTITURN_VALUE_POSITION=first_response



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

model=Qwen/Qwen3.5-4B
lr=4e-7
max_model_len=102400
max_new_tokens_per_turn=2048
val_batch_size=512
train_batch_size=8
num_chains=8
max_concurrent_chains=48
mini_batch_size=$((train_batch_size * num_chains))
sequence_parallel_size=8
kl_coef=0.001
train_dataset="./data/rlhf/os/r2e-gym-lite.json"
eval_dataset="./data/rlhf/os/r2e-gym-lite.json"
tools="[create_file,read_file,edit_file,grep_search,undo_edit,run_python]"
reward_name="r2e_gym_reward"
train_on_last_turn=False
base_model_name=$(basename $model)

# Long-context: use_remove_padding=True + ulysses_sequence_parallel_size=4 splits the sequence across
# 4 GPUs so activation memory per GPU is ~4x lower while keeping full max_model_len (e.g. 32768).
# If still OOM, try ulysses_sequence_parallel_size=8 (must divide n_gpus; 32/8=4 DP replicas).

# adv_estimator=rloo
# adv_estimator=reinforce_plus_plus
# adv_estimator=remax
adv_estimator=grpo
# adv_estimator=gae
# adv_estimator=sspo_multiturn
use_critic=False
critic_lr=5e-6
gamma=0.99
lam=0.95

entropy_coeff=0.001
kl_loss_type=mse
agent_type=qwen3coder_swe
max_turns=50
tool_parser_name="qwen3_coder"
total_training_steps=300
lr_warmup_steps_ratio=0.01
project_name="Resource"

# lora settings
lora_rank=32
lora_alpha=64
lora_target_modules="all-linear"

experiment_name="swe_r2e_gym_tools_${base_model_name}_${adv_estimator}_valueposition-${SSPO_MULTITURN_VALUE_POSITION}_lora_${lora_rank}"

python -m agentfly.cli train \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=${train_dataset} \
    data.val_files=${eval_dataset} \
    data.val_batch_size=$val_batch_size \
    data.train_batch_size=$train_batch_size \
    agent.use_agent=True \
    agent.init_config.agent_type=$agent_type \
    agent.init_config.model_name_or_path=$model \
    agent.init_config.max_model_len=$max_model_len \
    agent.init_config.tool_parser_name=$tool_parser_name \
    agent.init_config.tools=${tools} \
    agent.init_config.reward_name=${reward_name} \
    agent.run_config.generation_config.max_tokens=$max_new_tokens_per_turn \
    agent.run_config.max_turns=${max_turns} \
    agent.run_config.num_chains=$num_chains \
    agent.run_config.max_concurrent_chains=$max_concurrent_chains \
    agent.run_config.context_config.resource_backend=ray \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=$lora_rank \
    actor_rollout_ref.model.lora_alpha=$lora_alpha \
    actor_rollout_ref.model.lora.rank=$lora_rank \
    actor_rollout_ref.model.lora.alpha=$lora_alpha \
    actor_rollout_ref.model.use_fused_kernel=True \
    actor_rollout_ref.model.target_modules=$lora_target_modules \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$sequence_parallel_size \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$sequence_parallel_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.enable=$use_critic \
    critic.model.path=$model \
    critic.optim.lr=$critic_lr \
    critic.ppo_mini_batch_size=$train_batch_size \
    critic.model.use_remove_padding=True \
    critic.ulysses_sequence_parallel_size=$sequence_parallel_size \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.enable_activation_offload=False \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gamma \
    algorithm.lam=$lam \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${worker_num} \
    trainer.save_freq=20 \
    trainer.test_freq=${total_training_steps} \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False
