local_dir="/mnt/weka/home/renxi.wang/Research/AgentFly/checkpoints/Resource/swe_r2e_gym_tools_Qwen3.5-9B/global_step_100/actor"
target_dir="/mnt/weka/home/renxi.wang/Research/AgentFly/checkpoints/Resource/swe_r2e_gym_tools_Qwen3.5-9B/"
# local_dir="/mnt/weka/home/renxi.wang/Research/AgentFly/checkpoints/Resource/swe_r2e_gym_tools_Qwen3-32B_system/global_step_120/actor"
# target_dir="/mnt/weka/home/renxi.wang/Research/AgentFly/checkpoints/Resource/swe_r2e_gym_tools_Qwen3-32B_system/"
python -m agentfly.verl.model_merger merge --backend fsdp --local_dir ${local_dir} --target_dir ${target_dir}
