#!/bin/bash
#SBATCH --job-name=data_generation
#SBATCH --account=iq
#SBATCH --partition=main
#SBATCH --time=200:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
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
# Unbuffered so a crashing driver's last output and traceback reach the log (job 2631125
# died at exit code 120 during a checkpoint save with its final output lost in the buffer).
export PYTHONUNBUFFERED=1

# VLM judge ensemble: comma-separated lists, position-aligned (model[i] is served at ip[i]).
# JUDGES_ENV is the judges.env written by deploy_vlm_judges.sbatch once all judges are healthy.
if [[ -z "${JUDGES_ENV:-}" || ! -f "${JUDGES_ENV}" ]]; then
    echo "ERROR: set JUDGES_ENV to the judges.env written by deploy_vlm_judges.sbatch"; exit 1
fi
source "${JUDGES_ENV}"
echo "VLM_MODELS=${VLM_MODELS}"
echo "VLM_SERVER_IPS=${VLM_SERVER_IPS}"
for ip in $(echo "${VLM_SERVER_IPS}" | tr ",|" "  "); do  # TEMP (GLM replicas): also split "ip1|ip2"
    curl -sf --max-time 10 "http://${ip}:8000/health" >/dev/null || { echo "ERROR: judge ${ip}:8000 not healthy"; exit 1; }
done
# Rendered videos are read by the judges through file:// paths, so they must live on the
# shared filesystem (the judges allow local media under /mnt/weka).
export VLM_SHARED_VIDEO_DIR="$(pwd)/tmp/vlm_judge/${SLURM_JOB_ID}"
mkdir -p "${VLM_SHARED_VIDEO_DIR}"


# =================== Ray start ===================
# ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 uv run --no-sync ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
    uv run --no-sync ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
        uv run --no-sync ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &
done
sleep 10

# Policy model and template
# Copy of .../ViPhy/new_models/deepseek-r1-distill-qwen-7b_sft-multiple (weights symlinked)
# with tokenizer_class=PreTrainedTokenizerFast. transformers 5.x turns the original
# LlamaTokenizerFast class into a tokenizer that drops spaces/newlines (job 2616919: every
# prompt mis-tokenized, no ```python block found, all rewards 0).
# 32B: MODEL=.../ViPhy/new_models/qwen3-32b_sft-multiple TEMPLATE=qwen3-think-no-tool-no-system
# ROLLOUT_TP=4, submitted with sbatch --nodes=4.
model="${MODEL:-/mnt/weka/home/renxi.wang/.cache/simuscene_policy/deepseek-r1-distill-qwen-7b_sft-multiple}"

## Todo: add template
template="${TEMPLATE:-deepseek-r1-distill-qwen}"

# Data paths (must contain fields: prompt, vlm_questions)
train_file="./data/rlhf/simuphy/rl_set.json"
val_file="./data/rlhf/simuphy/val_set.json"
prompt_key="prompt"

lr=5e-7
max_model_len=16384
max_prompt_length=2048
max_new_tokens_per_turn=14336
batch_size=32
micro_batch_size_per_gpu=1
group_size=8
kl_coef=0.001
adv_estimator=grpo
mini_batch_size=128
entropy_coeff=0.001
kl_loss_type=mse

max_turns=1
project_name="SimuScene"
total_training_steps="${TOTAL_TRAINING_STEPS:-150}"

# Agent config
agent_type="hf"
tools="[answer_qa]"
# vlm_as_judge_pass_reward_multi_model (binary, video-level 2-of-3 vote) or
# vlm_as_judge_ratio_reward_multi_model (fraction of questions with a per-question 2-of-3 vote).
reward_name="${REWARD_NAME:-vlm_as_judge_pass_reward_multi_model}"

experiment_name="${EXPERIMENT_NAME:-simuscene_7b_sft-multiple_vlm-ensemble_rerun}"

# Run the driver with the project venv directly: under `uv run`, Ray's uv runtime-env hook
# rebuilds the env for workers and crashes on AgentFly's working_dir=None (job 2616900).
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
.venv/bin/python -m agentfly.cli train \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files=${train_file} \
    data.val_files=${val_file} \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    agent.init_config.agent_type=${agent_type} \
    agent.init_config.tools=${tools} \
    agent.init_config.template=${template} \
    agent.init_config.model_name_or_path=${model} \
    agent.init_config.reward_name=${reward_name} \
    agent.init_config.max_model_len=${max_model_len} \
    agent.run_config.generation_config.max_tokens=${max_new_tokens_per_turn} \
    agent.run_config.max_turns=${max_turns} \
    agent.run_config.group_size=${group_size} \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-2} \
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
    trainer.val_before_train=True
