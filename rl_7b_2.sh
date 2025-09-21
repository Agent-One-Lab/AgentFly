#!/bin/bash
#SBATCH --job-name=rl_7b_file2
#SBATCH --account=iq
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=720:00:00
#SBATCH --output=slurm/rl_7b_file2_%j.log
#SBATCH --error=slurm/rl_7b_file2_%j.err


export WANDB_API_KEY="817968bae37f1e87dcc478849b7c8a78a49e96a5"

conda activate ~/yxwang/envs/agentfly
cd ~/yxwang/AgentFly/verl

bash run_agents/run_code_vlm_as_judge_2.sh