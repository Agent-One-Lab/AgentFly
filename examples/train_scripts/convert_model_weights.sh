export PYTHONPATH=/mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/verl:$PYTHONPATH

python3 verl/scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/checkpoints/AgentRL/vlm_as_judge_reward_ladder_base_7b_deepseek_sft_single_sft-20260113161217/global_step_60/actor \
    --target_dir /mnt/weka/shrd/ad/haonan.li/ViPhy/new_models/rl_exploration/deepseek-r1-distill-qwen-7b_sft-single-rl_ladder_global_step_60

python3 verl/scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/checkpoints/AgentRL/vlm_as_judge_reward_ladder_base_7b_deepseek_sft_single_sft-20260113161217/global_step_80/actor \
    --target_dir /mnt/weka/shrd/ad/haonan.li/ViPhy/new_models/rl_exploration/deepseek-r1-distill-qwen-7b_sft-single-rl_ladder_global_step_80

python3 verl/scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/checkpoints/AgentRL/vlm_as_judge_reward_ladder_base_7b_deepseek_sft_single_sft-20260113161217/global_step_100/actor \
    --target_dir /mnt/weka/shrd/ad/haonan.li/ViPhy/new_models/rl_exploration/deepseek-r1-distill-qwen-7b_sft-single-rl_ladder_global_step_100
