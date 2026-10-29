export VLLM_USE_V1=1
# Run in single node
export VERL_LOGGING_LEVEL=INFO

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
ray start --head --node-ip-address="$head_node_ip" --port=$port  --num-cpus 192 --num-gpus 1

model=Qwen/Qwen2.5-3B-Instruct
template=qwen2.5
lr=5e-7
max_model_len=8192
train_batch_size=64        # prompts per training step
max_new_tokens_per_turn=512
group_size=8               # rollouts per prompt
num_gpus=1
# PPO batch sizes count trajectory rows (prompts x chains), not prompts. One mini-batch of all
# rows = one optimizer step per training step (fully on-policy). Each GPU's share of the
# mini-batch (ppo_mini_batch_size / num_gpus) must be a multiple of the micro-batch.
ppo_mini_batch_size=$((train_batch_size * group_size))
ppo_micro_batch_size_per_gpu=8

kl_coef=0.001
train_dataset="./data/rlhf/math/gsm8k_train.json"
val_dataset="./data/rlhf/math/gsm8k_test.json"
# adv_estimator=rloo
# adv_estimator=reinforce_plus_plus
# adv_estimator=remax
adv_estimator=grpo
# adv_estimator=gae

agent_type=hf
tools="[calculator]"
reward_name="math_equal_reward_tool"
entropy_coeff=0.001
kl_loss_type=mse
# Turns per episode: the model calls the calculator, sees the result, then answers. Leave room
# for that final answer: the reward grades the model's last message, not the tool output.
max_turns=3
system_prompt="You are a math assistant. Use the calculator tool for arithmetic. When you know the result, give the final answer as \\boxed{ANSWER}."
agent_backend="async_verl"
project_name="AgentRL"
total_training_steps=100

experiment_name="test_gsm8k"

python3 -m agentfly.cli train \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=$train_dataset \
    data.val_files=$val_dataset \
    data.train_batch_size=${train_batch_size} \
    agent.init_config.agent_type=$agent_type \
    agent.init_config.tools=$tools \
    agent.init_config.model_name_or_path=$model \
    agent.init_config.backend_config.backend=${agent_backend} \
    agent.init_config.reward_name=$reward_name \
    "agent.init_config.system_prompt='${system_prompt}'" \
    agent.init_config.max_model_len=$max_model_len \
    agent.run_config.generation_config.max_tokens=$max_new_tokens_per_turn \
    agent.run_config.max_turns=${max_turns} \
    agent.run_config.group_size=$group_size \
    agent.use_agent=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=$model \
    critic.ppo_mini_batch_size=${ppo_mini_batch_size} \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=True
