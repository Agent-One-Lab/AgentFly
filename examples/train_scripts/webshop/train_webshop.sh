model=Qwen/Qwen2.5-3B-Instruct


system_prompt="You are an autonomous shopping agent operating in the WebShop web environment. Your goal is to purchase exactly one product that best matches the user's natural-language instruction.
You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, you can do one action by <action> action </action>. If you think you have finished the task, summarize what you have done.

## WebShop page types (you will infer from observation)

- WebShop states are webpages of four types: Search page, Results page, Item page, Item-detail page.

## Available actions (the ONLY actions you may take)

### Search action (only available on the Search page):

- search[QUERY]: Search -> Results
- QUERY should be a short, high-signal shopping query.

### Choose action (only available on non-search pages; you must choose one of the clickable buttons shown in the page):

- choose[Back to search]: * -> Search
- choose[Prev page] / choose[Next page]: Results -> Results
- choose[PRODUCT_TITLE]: Results -> Item
- choose[OPTION]: Item -> Item (e.g., size/color/pack)
- choose[Desc] / choose[Overview]: Item -> Item-detail
- choose[Previous]: Item-detail -> Item
- choose[Buy] (or choose[Buy Now] if that is the button text shown): Item -> Episode End

Remember that you must put your action inside <action> and </action> tags."

template=action-agent
lr=5e-7
max_model_len=16384
max_new_tokens_per_turn=384
val_batch_size=512
batch_size=64
num_chains=8
# full on-policy
mini_batch_size=$((batch_size * num_chains))
kl_coef=0.001
train_dataset="./data/rlhf/webshop/webshop_goals_train.json"
eval_dataset="./data/rlhf/webshop/webshop_goals_val.json"
# adv_estimator=rloo
# adv_estimator=reinforce_plus_plus
# adv_estimator=remax
adv_estimator=grpo
# adv_estimator=gae

agent_type=action
tools="[webshop_browser_action]"
reward_name="webshop_reward"

entropy_coeff=0.001
kl_loss_type=mse
max_turns=15
lr_warmup_steps_ratio=0.01
total_training_steps=200

model_base_name=$(basename $model)
project_name="Open"
experiment_name="webshop_${model_base_name}_${adv_estimator}_test"

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
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=$model \
    critic.ppo_mini_batch_size=32 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
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
