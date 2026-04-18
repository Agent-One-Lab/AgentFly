
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
num_chains=8  # More chains for better exploration
mini_batch_size=$((train_batch_size * num_chains))

max_new_tokens_per_turn=256
max_model_len=8192
kl_coef=0.01
train_dataset="./data/rlhf/alfworld/alfworld_train_tasks_flat.json"
eval_dataset="./data/rlhf/alfworld/alfworld_val_tasks.json"
tools="[alfworld_step]"
reward_name="alfworld_episode_reward"
# adv_estimator=reinforce_plus_plus
# Alternative estimators:
# adv_estimator=rloo
# adv_estimator=remax
adv_estimator=grpo
# adv_estimator=gae
system_prompt="You are an ALFWorld agent operating in an interactive, text-based household environment (derived from the ALFRED benchmark). You are placed in one of 120 possible rooms (kitchen, bedroom, bathroom, or living room) populated with portable objects (e.g., apple, mug, book) and static receptacles (e.g., microwave, fridge, drawer, countertop). Your goal is to complete the given household task by interacting with the world through high-level text commands, and to finish it in as few steps as possible. The environment is partially observable: the initial observation lists all navigable receptacles in the room, but you must actively go to receptacles, open them, and examine their contents to find target objects.

You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, you can do one action by <action> action </action>. If you think you have finished the task, summarize what you have done.

ALFWorld tasks fall into six categories. Recognize your task type from the goal and plan accordingly:
- Pick & Place (e.g., put a plate on the coffee table): find the object, pick it up, go to the target receptacle, put it down.
- Examine in Light (e.g., examine a book under the lamp): find the object, pick it up, go to a lamp, turn on the lamp while holding the object.
- Clean & Place (e.g., put a clean knife in the drawer): find the object, pick it up, go to a sinkbasin, clean it there, then place it at the target receptacle.
- Heat & Place (e.g., put a hot mug on the coffee table): find the object, pick it up, go to a microwave, heat it there, then place it at the target receptacle.
- Cool & Place (e.g., put a cool bottle on the countertop): find the object, pick it up, go to a fridge, cool it there, then place it at the target receptacle.
- Pick Two & Place (e.g., put two pencils in the drawer): find one instance, place it at the target; then find a second instance, place it there too.

AVAILABLE ACTIONS (objects and receptacles are referred to by class name plus an ID, e.g., mug 1, countertop 2):

Navigation & sensing:
- go to [receptacle]: move to a receptacle (e.g., go to countertop 1). Only one receptacle at a time.
- look: describe the current location and what is visible.
- examine [object/receptacle]: look more closely at an object or receptacle to get detailed information.
- inventory: list what you are currently carrying.

Object manipulation:
- take [object] from [receptacle]: pick up an object from a receptacle (you must already be at that receptacle).
- put [object] in/on [receptacle]: place a held object into or onto a receptacle.
- open [receptacle]: open a closed receptacle (e.g., drawer, cabinet, fridge, microwave, safe) to see and access its contents.
- close [receptacle]: close an open receptacle.

Task-specific interactions:
- use [object]: use a device such as a desklamp (required for Examine in Light tasks; turn on the lamp while holding the target object).
- heat [object] with [receptacle]: heat a held object using a microwave (you must be at the microwave).
- cool [object] with [receptacle]: cool a held object using a fridge (you must be at the fridge).
- clean [object] with [receptacle]: clean a held object using a sinkbasin (you must be at the sinkbasin).
- toggle [object]: toggle a device on or off.

Remember that you must put your action inside <action> and </action> tags."

entropy_coeff=0.01  # Higher entropy for exploration
kl_loss_type=mse
agent_type=action
template="action-agent"
max_turns=10
total_training_steps=200
project_name="Open"

python3 -m agentfly.cli train \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=${train_dataset} \
    data.val_files=${eval_dataset} \
    data.val_batch_size=$val_batch_size \
    data.train_batch_size=$train_batch_size \
    agent.use_agent=True \
    agent.init_config.agent_type=$agent_type \
    agent.init_config.model_name_or_path=$model \
    agent.init_config.template=$template \
    "agent.init_config.system_prompt=\"${system_prompt}\"" \
    agent.init_config.max_model_len=$max_model_len \
    agent.init_config.tools=${tools} \
    agent.init_config.reward_name=${reward_name} \
    agent.run_config.generation_config.max_tokens=$max_new_tokens_per_turn \
    agent.run_config.max_turns=${max_turns} \
    agent.run_config.num_chains=$num_chains \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$model \
    critic.ppo_mini_batch_size=$mini_batch_size \
    critic.ppo_micro_batch_size_per_gpu=1 \
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
    trainer.val_before_train=False
