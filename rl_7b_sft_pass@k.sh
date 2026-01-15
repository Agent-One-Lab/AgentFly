#!/bin/bash
#SBATCH --job-name=rl_7b_sft_pass@k
#SBATCH --account=iq
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=720:00:00
#SBATCH --output=slurm/rl_7b_sft_pass@k_%j.log
#SBATCH --error=slurm/rl_7b_sft_pass@k_%j.err

export WANDB_API_KEY='817968bae37f1e87dcc478849b7c8a78a49e96a5'
export PYTHONPATH=/mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/verl:$PYTHONPATH
conda activate Agentfly
export TORCH_LIBDIR="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIBDIR:$LD_LIBRARY_PATH"
cd /mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly

bash examples/train_scripts/run_code_vlm_as_judge_reward_pass@k_base_7b_deepseek_sft_single_sft.sh