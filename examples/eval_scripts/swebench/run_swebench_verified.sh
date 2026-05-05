# Directory holding the swe-bench-verified enroot images.
# See docs/examples/swe.md for how to fetch the images and set this var.
export ENROOT_IMAGES_PATH="${ENROOT_IMAGES_PATH:-/path/to/enroot/images/swe-bench-verified}"

# model_name_or_path="Qwen/Qwen3.5-4B"
# model_name_or_path="Qwen/Qwen3.5-9B"

model_name_or_path="Qwen/Qwen3-32B"

base_name=$(basename $model_name_or_path)

agent="bash"
setting="bash"
result_dir="results/swe/${base_name}-agent_${agent}-setting_${setting}"
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
  --agent ${agent} \
  --tool-set ${setting} \
  # --backend async_vllm \
  # --tp 2 --dp 4