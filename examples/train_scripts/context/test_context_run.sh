
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
# Reward decomposition strategy: last_only | broadcast | uniform | geometric
export REWARD_DECOMPOSITION=uniform
# Decay rate for geometric mode; ignored otherwise.
export REWARD_DECOMPOSITION_GAMMA=0.9
# Trigger summarize after this many assistant turns
export CONTEXT_TRIGGER_TURNS=10
# Context trigger message variant: base | detail
export CONTEXT_TRIGGER_MESSAGE_TYPE=base

export CONTEXTRL_SPPO_VALUE_POSITION=first_response

# export VERL_LOGGING_LEVEL=DEBUG

# Remove existing Ray cluster
ray stop
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
ray start --head --node-ip-address="$head_node_ip" --port=$port  --num-cpus 192 --num-gpus 8


model=Qwen/Qwen3-4B-Instruct-2507

# system_prompt="You are a ScienceWorld agent operating in an interactive, text-based environment that simulates elementary-school science tasks (e.g., thermodynamics, simple circuits, chemistry, biology). Your goal is to complete the current task by interacting with the world through text commands, earning the highest possible task score, and finishing efficiently. The environment is partially observable; you must actively examine rooms, containers, and your inventory to gather needed information.
# You must conduct reasoning inside <think> and </think> first every time you get new information.

# AVAILABLE ACTIONS (you may use these; some take 0, 1, or 2 arguments):
# Core navigation & sensing:
# - go to [location]: move to a new location
# - look around: describe the current room
# - look at [object]: describe an object in detail
# - look in [container]: describe a container's contents
# - read [object]: read a note or book
# - focus on [object]: signal intent on a task object
# - task: describe current task
# - inventory: list agent's inventory
# - wait [duration]: take no action for some duration

# Object manipulation:
# - pick up [object]: move an object to the inventory
# - put down [object]: drop an inventory item
# - move [object] to [container]: move an object to a container
# - open [object]: open a container
# - close [object]: close a container
# - activate [object]: activate a device
# - deactivate [object]: deactivate a device
# - use [tool] [on [object]]: use a device/item

# Liquids & chemistry:
# - pour [liquid/container] into [container]: pour a liquid into a container
# - dunk [container] into [liquid]: dunk a container into a liquid
# - mix [container]: chemically mix a container

# Living things / misc:
# - eat [object]: eat a food
# - flush [object]: flush a toilet

# Electricity (for simple circuits):
# - connect [object] to [object]: connect electrical components
# - disconnect [object]: disconnect electrical components

# For these actions, you must enclose them with <action> action </action>.

# Additionally, you can call a summarization tool to summarize all the information you have. For summarization, you must enclose it with <summarize> your summary here </summarize>.

# - Summarization should be used to help you complete the task in future steps. Do not give any conclusion on whether the task has been or can not be finished.

# **Do not repeat existing information.** If you think you have finished the task, don't do any action or call any tool, directly describe what has been done."

system_prompt_context="You are a ScienceWorld agent operating in an interactive, text-based environment that simulates elementary-school science tasks (e.g., thermodynamics, simple circuits, chemistry, biology). Your goal is to complete the current task by interacting with the world through text commands, earning the highest possible task score, and finishing efficiently. The environment is partially observable; you must actively examine rooms, containers, and your inventory to gather needed information.

Before any action, you must conduct reasoning inside <think> and </think>.

AVAILABLE ACTIONS (you may use these; some take 0, 1, or 2 arguments):
Core navigation & sensing:
- go to [location]: move to a new location
- look around: describe the current room
- look at [object]: describe an object in detail
- look in [container]: describe a container's contents
- read [object]: read a note or book
- focus on [object]: signal intent on a task object
- task: describe current task
- inventory: list agent's inventory
- wait [duration]: take no action for some duration

Object manipulation:
- pick up [object]: move an object to the inventory
- put down [object]: drop an inventory item
- move [object] to [container]: move an object to a container
- open [object]: open a container
- close [object]: close a container
- activate [object]: activate a device
- deactivate [object]: deactivate a device
- use [tool] [on [object]]: use a device/item

Liquids & chemistry:
- pour [liquid/container] into [container]: pour a liquid into a container
- dunk [container] into [liquid]: dunk a container into a liquid
- mix [container]: chemically mix a container

Living things / misc:
- eat [object]: eat a food
- flush [object]: flush a toilet

Electricity (for simple circuits):
- connect [object] to [object]: connect electrical components
- disconnect [object]: disconnect electrical components

For these actions, you must enclose them with <action> action </action>.
 
After each stage of exploring, you must call a summarization tool to summarize all the information you have. For summarization, you must enclose it with <summarize> your summary here </summarize>.

- After you have summarized, your summary will be in [Previous Summary]. Then a new stage starts.

- Every task is solvable. Try to explore the world as much as possible.

- Don't call summarization before taking any action.

- If you think you have completed the task successfully, put the phrase end task inside <summarize> and </summarize>: <summarize> end task </summarize>"

system_prompt="You are a ScienceWorld agent operating in an interactive, text-based environment that simulates elementary-school science tasks (e.g., thermodynamics, simple circuits, chemistry, biology). Your goal is to complete the current task by interacting with the world through text commands, earning the highest possible task score, and finishing efficiently. The environment is partially observable; you must actively examine rooms, containers, and your inventory to gather needed information.
You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, you can do one action by <action> action </action>. If you think you have finished the task, summarize what you have done.

AVAILABLE ACTIONS (you may use these; some take 0, 1, or 2 arguments):
Core navigation & sensing:
- go to [location]: move to a new location
- look around: describe the current room
- look at [object]: describe an object in detail
- look in [container]: describe a container's contents
- read [object]: read a note or book
- focus on [object]: signal intent on a task object
- task: describe current task
- inventory: list agent's inventory
- wait [duration]: take no action for some duration

Object manipulation:
- pick up [object]: move an object to the inventory
- put down [object]: drop an inventory item
- move [object] to [container]: move an object to a container
- open [object]: open a container
- close [object]: close a container
- activate [object]: activate a device
- deactivate [object]: deactivate a device
- use [tool] [on [object]]: use a device/item

Liquids & chemistry:
- pour [liquid/container] into [container]: pour a liquid into a container
- dunk [container] into [liquid]: dunk a container into a liquid
- mix [container]: chemically mix a container

Living things / misc:
- eat [object]: eat a food
- flush [object]: flush a toilet

Electricity (for simple circuits):
- connect [object] to [object]: connect electrical components
- disconnect [object]: disconnect electrical components

Remember that you must put your action inside <action> and </action> tags."

template=action-agent
lr=4e-7
max_model_len=8192
max_new_tokens_per_turn=512
val_batch_size=512
batch_size=32
num_chains=4
# full on-policy
mini_batch_size=$((batch_size * num_chains))
kl_coef=0.001
train_dataset="./data/rlhf/scienceworld/scienceworld_train.json"
eval_dataset="./data/rlhf/scienceworld/scienceworld_test.json"
# adv_estimator=rloo
# adv_estimator=reinforce_plus_plus
# adv_estimator=remax
# adv_estimator=grpo
# adv_estimator=gae
# adv_estimator=contextrl
# use_critic=True
# adv_estimator=contextrl_depth_grouped

adv_estimator=contextrl_sppo
use_critic=True

critic_lr="5e-6"

agent_type=action
tools="[scienceworld_explorer,summarize]"
reward_name="scienceworld_reward"

entropy_coeff=0.001
kl_loss_type=mse
max_turns=30
lr_warmup_steps_ratio=0.01
total_training_steps=300
gamma=0.99
lam=0.95

model_base_name=$(basename $model)
project_name="Context"
experiment_name="scienceworld_${model_base_name}_${adv_estimator}_fix_message-${CONTEXT_TRIGGER_MESSAGE_TYPE}_triggerturns-${CONTEXT_TRIGGER_TURNS}_decomp-${REWARD_DECOMPOSITION}_compare_valueposition-${CONTEXTRL_SPPO_VALUE_POSITION}"

python -m agentfly.cli train \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=${train_dataset} \
    data.val_files=${eval_dataset} \
    data.val_batch_size=$val_batch_size \
    data.train_batch_size=$batch_size \
    agent.use_agent=True \
    agent.init_config.agent_type=$agent_type \
    "agent.init_config.system_prompt=\"${system_prompt}\"" \
    agent.init_config.max_model_len=$max_model_len \
    agent.init_config.tools=$tools \
    agent.init_config.template=$template \
    agent.init_config.model_name_or_path=$model \
    agent.init_config.reward_name=$reward_name \
    agent.run_config.generation_config.max_tokens=$max_new_tokens_per_turn \
    agent.run_config.max_turns=${max_turns} \
    agent.run_config.num_chains=$num_chains \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.enable=$use_critic \
    critic.model.path=$model \
    critic.optim.lr=$critic_lr \
    critic.ppo_mini_batch_size=32 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gamma \
    algorithm.lam=$lam \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=300 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False
