#!/bin/bash
#SBATCH --job-name=rl_base_7b
#SBATCH --account=iq
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=720:00:00
#SBATCH --output=slurm/rl_base_7b_%j.log
#SBATCH --error=slurm/rl_base_7b_%j.err


export WANDB_API_KEY='817968bae37f1e87dcc478849b7c8a78a49e96a5'
conda activate Agentfly
cd /mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/verl

bash run_agents/run_code_vlm_as_judge_base_7b.sh