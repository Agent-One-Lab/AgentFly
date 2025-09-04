Training Example
==============

Finally, we are ready to train the agent.

**1. Prepare Training Data**

----------------

We show an example of training on GSM8K dataset. First, prepare your training and validation datasets in JSON format. The datasets should follow this structure:

```

[
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "10"
    },
    ...
]
```

We use `question` filed to put task queries, and these "questions" will be used to form input messages. While other fileds, in our case, "answer" will be given to the reward function.

**2. Create Training Script**

------------------------
Create a training script (e.g., ``train_example.sh``) with the following configuration:

```bash

export WANDB_API_KEY="your_wandb_key"  # For logging to Weights & Biases
export VLLM_USE_V1=1
# Run in single node

set -x

export head_node=${nodes[0]}

head_node_ip=$(hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export HYDRA_FULL_ERROR=1
# Remove existing Ray cluster
ray stop
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
ray start --head --node-ip-address="$head_node_ip" --port=$port  --num-cpus 192 --num-gpus 1

model=Qwen/Qwen2.5-3B-Instruct
template=qwen2.5
lr=5e-7
length=256
batch_size=32
num_chains=8
kl_coef=0.001
train_dataset="./data/rlhf/math/gsm8k_train.json"
val_dataset="./data/rlhf/math/gsm8k_test.json"
# adv_estimator=rloo
# adv_estimator=reinforce_plus_plus
# adv_estimator=remax
adv_estimator=grpo
# adv_estimator=gae

mini_batch_size=$batch_size

agent_type=hf
tools="[calculator]"
reward_name="math_reward_string_equal"
entropy_coeff=0.001
kl_loss_type=mse
max_turns=3
agent_backend="async_verl"
project_name="AgentRL"
total_training_steps=200

experiment_name="test_gsm8k"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=$train_dataset \
    data.val_files=$val_dataset \
    data.train_batch_size=$batch_size \
    agent.agent_type=$agent_type \
    agent.tools=$tools \
    agent.template=$template \
    agent.model_name_or_path=$model \
    agent.max_turns=${max_turns} \
    agent.backend=${agent_backend} \
    agent.reward_name=$reward_name \
    agent.num_chains=$num_chains \
    agent.use_agent=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
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
    actor_rollout_ref.rollout.response_length=$length \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=$model \
    critic.ppo_mini_batch_size=${mini_batch_size} \
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
    trainer.val_before_train=False
```

**3. Run Training**

--------------

Execute the training script. This training script run agent RL in a single node with one GPU. We have wrapped up everything, including tools, rewards, and training data. Run the following command to start training.

```
cd verl
bash run_agents/train_example.sh
```

The training progress will be logged to Weights & Biases if configured. You can monitor metrics like reward, loss, and KL divergence during training.

Key parameters to consider:

- ``model``: Base model to fine-tune
- ``batch_size``: Training batch size
- ``lr``: Learning rate
- ``num_chains``: Number of interaction chains per sample
- ``max_turns``: Maximum turns per interaction chain
- ``total_training_steps``: Total number of training steps
