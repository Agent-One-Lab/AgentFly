export VLLM_USE_V1=1
# Run in single node
export VERL_LOGGING_LEVEL=DEBUG
# export PYTHONPATH="agentfly/verl:${PYTHONPATH}"

set -x

export head_node=${nodes[0]}

head_node_ip=$(hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

# export VLLM_ATTENTION_BACKEND=XFORMERS
# export GLOO_SOCKET_IFNAME=ens10f0np0
export HYDRA_FULL_ERROR=1
# Remove existing Ray cluster
ray stop
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
ray start --head --node-ip-address="$head_node_ip" --port=$port  --num-cpus 192 --num-gpus 8


# Policy model and template
model="/mnt/weka/shrd/ad/haonan.li/ViPhy/new_models/deepseek-r1-distill-qwen-7b_sft-multiple"
## Todo: add template
template="deepseek-r1-distill-qwen"

# Data paths (must contain fields: question, vlm_questions)
train_file="/mnt/weka/home/yongxin.wang/workspace/human_modify/rl_set.json"
val_file="/mnt/weka/home/yongxin.wang/workspace/human_modify/val_set.json"

lr=5e-7
length=16384
max_prompt_length=2048
max_new_tokens_per_turn=14336
batch_size=128
micro_batch_size_per_gpu=1
num_chains=8
kl_coef=0.001
adv_estimator=grpo
mini_batch_size=32
entropy_coeff=0.001
kl_loss_type=mse

max_turns=1
agent_backend="async_verl"
project_name="AgentRL"
total_training_steps=2000

# Agent config
agent_type="hf"
tools=""
reward_name="vlm_as_judge_reward_multi_model"
# SYSTEM_PROMPT="\\nYou are an expert computational physicist specializing in scientific visualization and simulation. \\nYou also an excellent programmer.\\nYour expertise includes creating educational physics simulations that effectively communicate complex physical phenomena to diverse audiences.\\n"

experiment_name="vlm_as_judge_reward_multi_model_base_7b_deepseek_sft_multiple"

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
    trainer.experiment_name="${experiment_name}-${lr}-${length}-bs${batch_size}-n${num_chains}-kl${kl_loss_type}${kl_coef}-entropy${entropy_coeff}-${max_turns}turns-${adv_estimator}-${TIMESTAMP}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${worker_num} \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False
