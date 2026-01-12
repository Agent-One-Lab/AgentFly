hf_model_path="Qwen/Qwen2.5-3B-Instruct"
local_dir="/mnt/weka/home/renxi.wang/Agent-One-Lab/AgentFly/verl/checkpoints/AgentRL/webshop_agent/global_step_50/actor"
target_dir="/mnt/weka/home/renxi.wang/Agent-One-Lab/AgentFly/verl/checkpoints/AgentRL/webshop_agent/"
python scripts/model_merger.py --backend fsdp --hf_model_path $hf_model_path --local_dir ${local_dir} --target_dir ${target_dir}