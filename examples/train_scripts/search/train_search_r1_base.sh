
# Run in single node

set -x

export head_node=${nodes[0]}

head_node_ip=$(hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

# export VLLM_ATTENTION_BACKEND=XFORMERS
# export GLOO_SOCKET_IFNAME=ens10f0np0
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

# export VERL_LOGGING_LEVEL=DEBUG

# Remove existing Ray cluster
ray stop
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
ray start --head --node-ip-address="$head_node_ip" --port=$port  --num-cpus 192 --num-gpus 8


model=Qwen/Qwen2.5-3B
lr=5e-7
max_model_len=16384
max_new_tokens_per_turn=378
val_batch_size=512
train_batch_size=256
num_chains=5
mini_batch_size=$((train_batch_size * num_chains))
kl_coef=0.001
train_dataset="./data/rlhf/qa/nq_hotpotqa_train.json"
eval_dataset="./data/rlhf/qa/qa_test.json"
system_prompt="Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."
# tools="[google_search,answer_qa]"
tools="[async_dense_retrieve_api]"
# tools="[dense_retrieve,answer_qa]"
# reward_name="qa_f1_reward"
# reward_name="qa_em_format_reward"
# experiment_name="search_r1_base_grpo_em_format"
reward_name="qa_em_reward"
train_on_last_turn=False
experiment_name="search_r1_base_grpo_em_reward_searchr1_setting_3_turns_shifting"

# adv_estimator=rloo
# adv_estimator=reinforce_plus_plus
# adv_estimator=remax
adv_estimator=grpo
# adv_estimator=gae
entropy_coeff=0.001
kl_loss_type=low_var_kl
agent_type=searchr1
max_turns=3
template="search-r1"
tool_parser_name="hermes"
total_training_steps=1005
lr_warmup_steps_ratio=0.03
project_name="Algorithm"

python3 -m agentfly.cli train \
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
    "agent.init_config.system_prompt=\"${system_prompt}\"" \
    agent.init_config.max_model_len=$max_model_len \
    agent.init_config.tool_parser_name=$tool_parser_name \
    agent.init_config.tools=${tools} \
    agent.init_config.reward_name=${reward_name} \
    agent.generation_config.max_tokens=$max_new_tokens_per_turn \
    agent.max_turns=${max_turns} \
    agent.num_chains=$num_chains \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
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
    critic.ppo_mini_batch_size=$train_batch_size \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=200 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False
