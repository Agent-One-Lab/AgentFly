
# Run in single node
export VLLM_USE_V1=1


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

# AlfWorld Configuration
model=Qwen/Qwen2.5-3B-Instruct
lr=1e-6
length=512  # Increased for longer episodes
val_batch_size=256
train_batch_size=64
num_chains=4  # More chains for better exploration
kl_coef=0.01
train_dataset="./data/rlhf/alfworld/train_tasks_flat.json"
eval_dataset="./data/rlhf/alfworld/val_tasks_flat.json"
tools="[alfworld_step,alfworld_get_admissible_commands,alfworld_get_task_objective]"
reward_name="alfworld_episode_reward"
adv_estimator=reinforce_plus_plus
# Alternative estimators:
# adv_estimator=rloo
# adv_estimator=remax
# adv_estimator=grpo
# adv_estimator=gae
task_info="Navigate the ALFWorld environment and complete tasks by interacting with objects. Use the tools provided to step through the environment. when you keep getting Nothing happens as a feedback use admissible commands to see why becasue it's very likely your action / command is wrong !!!"

entropy_coeff=0.01  # Higher entropy for exploration
kl_loss_type=mse
agent_type=react
max_turns=10  
prompt_template="qwen2.5-no-system-tool"
total_training_steps=150 
project_name="AgentFly-AlfWorld-RPP-1024"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=${train_dataset} \
    data.val_files=${eval_dataset} \
    agent.num_chains=$num_chains \
    data.val_batch_size=$val_batch_size \
    data.train_batch_size=$train_batch_size \
    agent.use_agent=True \
    agent.model_name_or_path=$model \
    agent.max_turns=${max_turns} \
    agent.agent_type=$agent_type \
    agent.tools=${tools} \
    agent.reward_name=${reward_name} \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=$length \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$model \
    critic.ppo_mini_batch_size=32 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name="${model}-alfworld-${lr}-${length}-bs${train_batch_size}-n${num_chains}-kl${kl_loss_type}${kl_coef}-entropy${entropy_coeff}-${max_turns}turns-${adv_estimator}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=True