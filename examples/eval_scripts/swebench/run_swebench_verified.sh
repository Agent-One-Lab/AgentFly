export ENROOT_IMAGES_PATH="/mnt/weka/home/renxi.wang/Agent-One-Lab/enroot-py/data/images/swe-bench-verified"

# model_name_or_path="Qwen/Qwen3.5-4B"
model_name_or_path="/mnt/weka/home/renxi.wang/Research/AgentFly/checkpoints/Resource/swe_r2e_gym_tools_Qwen3.5-9B"

# model_name_or_path="Qwen/Qwen3.5-9B"

# model_name_or_path="Qwen/Qwen3-32B"

# model_name_or_path="/mnt/weka/home/renxi.wang/Research/AgentFly/checkpoints/Resource/swe_r2e_gym_tools_Qwen3-32B_system"
base_name=$(basename $model_name_or_path)
result_dir="results/swe/${base_name}"
max_concurrent_chains=48
max_model_len=40000

python -m agentfly.cli swebench \
  --data-path data/rlhf/os/swe-bench-verified.json \
  --result-dir $result_dir \
  --model-name-or-path $model_name_or_path \
  --max-turns 30 \
  --temperature 0.0 \
  --reward-name r2e_gym_reward \
  --max-concurrent-chains ${max_concurrent_chains} \
  --max-model-len ${max_model_len} \
  --agent qwen3_coder \
  --tool-set file
#   --backend async_vllm \
#   --tp 2 --dp 4