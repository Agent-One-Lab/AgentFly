
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


model=Qwen/Qwen2.5-VL-3B-Instruct
lr=5e-7
length=512
val_batch_size=512
train_batch_size=512
num_chains=1
kl_coef=0.001

train_dataset="./data/rlhf/qa/infoseek_train.json"
eval_dataset="./data/rlhf/qa/infoseek_val.json"
reward_name="infoseek_reward"
tools="[asyncdense_retrieve,answer_qa]"
max_turns=5
experiment_name="vlm_search_agent"

# OK-VQA
# train_dataset="./data/rlhf/qa/OK-VQA.json"
# eval_dataset="./data/rlhf/qa/OK-VQA.json"
# reward_name="ok_vqa_reward"
# tools="[answer_qa]"
# max_turns=1
# experiment_name="vlm_qa"

# tools="[google_search,answer_qa]"
# tools="[dense_retrieve,answer_qa]"
# reward_name="qa_f1_reward"
# reward_name="qa_f1_reward_format"
# adv_estimator=rloo
adv_estimator=reinforce_plus_plus
# adv_estimator=remax
# adv_estimator=grpo
# adv_estimator=gae

entropy_coeff=0.001
kl_loss_type=mse
agent_type=react
template="qwen2.5-vl"
total_training_steps=200
project_name="Open"
trust_remote_code=True


# experiment_name="${model}-${train_dataset}-${lr}-${length}-bs${batch_size}-n${num_chains}-kl${kl_loss_type}${kl_coef}-entropy${entropy_coeff}-${max_turns}turns-${adv_estimator}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=${train_dataset} \
    data.val_files=${eval_dataset} \
    agent.num_chains=$num_chains \
    data.val_batch_size=$val_batch_size \
    data.train_batch_size=$train_batch_size \
    agent.use_agent=True \
    agent.model_name_or_path=$model \
    agent.template=$template \
    agent.max_turns=${max_turns} \
    agent.agent_type=$agent_type \
    agent.tools=${tools} \
    agent.reward_name=${reward_name} \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=$trust_remote_code \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=$length \
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
    trainer.test_freq=2000 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False
