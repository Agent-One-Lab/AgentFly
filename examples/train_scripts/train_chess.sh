export VLLM_USE_V1=1
export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="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"

set -x

head_node_ip=$(hostname --ip-address)
port=6379

# Remove existing Ray cluster
ray stop
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 96 --num-gpus 8

model="Qwen/Qwen2.5-3B-Instruct"
template="qwen2.5-no-system-tool"
lr=4e-7
max_model_len=8192
mini_batch_size=32
max_new_tokens_per_turn=256
num_chains=8
ppo_micro_batch_size_per_gpu=1


system_prompt="Solve the given chess puzzle. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, you can use the following tools to interact with the board:\n\n- <chess_get_state> </chess_get_state>: Get the current board state including FEN, visual board, whose turn it is, and puzzle status.\n- <chess_get_legal_moves> </chess_get_legal_moves>: List all legal moves in the current position.\n- <chess_move> move </chess_move>: Make a move on the board. The move can be in UCI format (e.g., 'e2e4') or standard algebraic notation (e.g., 'Nf3').\n\nYou should start by calling <chess_get_state> </chess_get_state> to see the current position. You may call <chess_get_legal_moves> </chess_get_legal_moves> to see all available moves. When you have decided on the best move, play it with <chess_move> move </chess_move>. Continue until the puzzle is solved. When the puzzle is solved, provide your final answer inside <answer> and </answer> listing the moves you played. For example, <answer> e2e4 d7d5 </answer>."

kl_coef=0.001
train_dataset="./data/chess/chess_puzzles_train.json"
val_dataset="./data/chess/chess_puzzles_val.json"
adv_estimator=grpo

agent_type="chess"
tools="[chess_move,chess_get_state,chess_get_legal_moves]"
reward_name="chess_puzzle_reward"
entropy_coeff=0.001
kl_loss_type=mse
max_turns=10
agent_backend="async_verl"
project_name="AgentRL"
total_training_steps=200

experiment_name="chess_puzzle_solver_$(date +%Y%m%d_%H%M%S)"

python3 -m agentfly.cli train \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=$train_dataset \
    data.val_files=$val_dataset \
    data.train_batch_size=${mini_batch_size} \
    agent.init_config.agent_type=$agent_type \
    agent.init_config.tools=$tools \
    agent.init_config.model_name_or_path=$model \
    agent.init_config.backend=${agent_backend} \
    agent.init_config.reward_name=$reward_name \
    agent.init_config.max_model_len=$max_model_len \
    agent.init_config.template=$template\
    agent.generation_config.max_tokens=$max_new_tokens_per_turn \
    agent.max_turns=${max_turns} \
    agent.num_chains=$num_chains \
    agent.use_agent=True \
    "agent.init_config.system_prompt=\"${system_prompt}\"" \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=$max_new_tokens_per_turn \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$model \
    critic.ppo_mini_batch_size=${mini_batch_size} \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False
