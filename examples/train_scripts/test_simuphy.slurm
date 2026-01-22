#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=200:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCh --account=iq
#SBATCH --cpus-per-task=96
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

# Policy model and template
model="/mnt/weka/shrd/ad/haonan.li/ViPhy/new_models/deepseek-r1-distill-qwen-7b_sft-multiple"
## Todo: add template
template="deepseek-r1-distill-qwen"

# Data paths (must contain fields: question, vlm_questions)
train_file="data/rlhf/simuphy/rl_set.json"
val_file="data/rlhf/simuphy/val_set.json"

lr=5e-7
max_model_len=16384
max_prompt_length=2048
max_new_tokens_per_turn=14336
batch_size=128
micro_batch_size_per_gpu=1
num_chains=8
kl_coef=0.001
adv_estimator=grpo
mini_batch_size=128
entropy_coeff=0.001
kl_loss_type=mse

max_turns=1
agent_backend="async_verl"
project_name="AgentRL"
total_training_steps=2000
tools="[answer_qa]"
# Agent config
agent_type="hf"
reward_name="vlm_as_judge_reward_multi_model"
# SYSTEM_PROMPT="\\nYou are an expert computational physicist specializing in scientific visualization and simulation. \\nYou also an excellent programmer.\\nYour expertise includes creating educational physics simulations that effectively communicate complex physical phenomena to diverse audiences.\\n"

# experiment_name="vlm_as_judge_reward_multi_model_base_7b_deepseek_sft_multiple"

experiment_name="test_simuphy_1"

python3 -m agentfly.cli train \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files=${train_file} \
    data.val_files=${val_file} \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    agent.init_config.agent_type=${agent_type} \
    agent.init_config.tools=${tools} \
    agent.init_config.template=${template} \
    agent.init_config.model_name_or_path=${model} \
    agent.init_config.backend=${agent_backend} \
    agent.init_config.reward_name=${reward_name} \
    agent.init_config.max_model_len=${max_model_len} \
    agent.generation_config.max_tokens=${max_new_tokens_per_turn} \
    agent.max_turns=${max_turns} \
    agent.num_chains=${num_chains} \
    agent.use_agent=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=${model} \
    critic.ppo_mini_batch_size=${mini_batch_size} \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${worker_num} \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False
