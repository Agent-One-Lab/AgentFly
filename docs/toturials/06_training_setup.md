# Tutorial 6: Training Setup

This tutorial covers how to set up and start training your customized AgentFly agents using reinforcement learning. You'll learn about configuration, distributed training, monitoring, and troubleshooting.

## Table of Contents

1. [Training Overview](#training-overview)
2. [Environment Setup](#environment-setup)
3. [Configuration](#configuration)
4. [Training Scripts](#training-scripts)
5. [Distributed Training](#distributed-training)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Troubleshooting](#troubleshooting)

## Training Overview

AgentFly training uses the VERL (Volcengine Reinforcement Learning) framework and supports multiple RL algorithms:

- **PPO (Proximal Policy Optimization)**: Standard policy gradient method
- **GRPO (Group Relative Policy Optimization)**: Efficient batch-based training
- **Reinforce++**: Enhanced REINFORCE with variance reduction
- **RLOO (Reinforcement Learning with Likelihood-ratio Objective Optimization)**: Alternative objective
- **ReMax**: Maximum entropy reinforcement learning

### Training Pipeline

1. **Data Loading**: Load training and validation datasets
2. **Agent Initialization**: Set up agent with tools and rewards
3. **Rollout Generation**: Generate agent trajectories
4. **Reward Calculation**: Compute rewards for trajectories
5. **Policy Update**: Update model parameters using RL algorithm
6. **Evaluation**: Test on validation set
7. **Model Saving**: Save checkpoints and final model

## Environment Setup

### Prerequisites

```bash
# Install AgentFly with VERL support
pip install -e .
pip install -e '.[verl]' --no-build-isolation

# Optional: Install Redis for search caching
conda install conda-forge::redis-server==7.4.0

# Optional: Install enroot for containerized tools
# See: https://github.com/NVIDIA/enroot/blob/master/doc/installation.md
```

### Ray Setup for Distributed Training

```bash
# Initialize Ray cluster (single node)
export VLLM_USE_V1=1
ray stop
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
head_node_ip=$(hostname --ip-address)
port=6379
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 192 --num-gpus 8
```

### Directory Structure

```bash
# Create training directory structure
mkdir -p training_project/{data,configs,scripts,outputs,logs}

training_project/
├── data/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── configs/
│   ├── agent_config.yaml
│   └── training_config.yaml
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
├── outputs/
│   └── checkpoints/
└── logs/
    └── training.log
```

## Configuration

### Agent Configuration

Create `configs/agent_config.yaml`:

```yaml
# Agent Configuration
agent:
  agent_type: "react"  # or "code", "custom"
  model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
  template: "qwen2.5-no-tool"
  backend: "async_verl"
  
  # Tools
  tools: 
    - "code_interpreter"
    - "google_search_serper"
    - "answer"
  
  # Agent behavior
  max_steps: 8
  num_chains: 8
  use_agent: true
  
  # Reward function
  reward_name: "math_reward_tool"
  
  # System prompt
  system_prompt: "You are a helpful assistant that thinks step by step and uses tools when needed."

# Model configuration
model:
  path: "Qwen/Qwen2.5-7B-Instruct"
  use_remove_padding: false
  enable_gradient_checkpointing: false
  
# Training data
data:
  train_files: "./data/train.json"
  val_files: "./data/val.json"
  train_batch_size: 64
  val_batch_size: 32

# Rollout configuration
rollout:
  name: "vllm"
  response_length: 512
  tensor_model_parallel_size: 2
  gpu_memory_utilization: 0.5
  log_prob_micro_batch_size_per_gpu: 4
```

### Training Configuration

Create `configs/training_config.yaml`:

```yaml
# Algorithm configuration
algorithm:
  adv_estimator: "grpo"  # or "ppo", "reinforce_plus_plus", "rloo", "remax"
  kl_ctrl:
    kl_coef: 0.001

# Actor-Critic configuration
actor_rollout_ref:
  actor:
    optim:
      lr: 5e-7
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 2
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: "mse"
    entropy_coeff: 0.001
    fsdp_config:
      param_offload: true
      optimizer_offload: true
  
  ref:
    log_prob_micro_batch_size_per_gpu: 4
    fsdp_config:
      param_offload: true

# Critic configuration
critic:
  model:
    path: "Qwen/Qwen2.5-7B-Instruct"
  ppo_mini_batch_size: 64
  ppo_micro_batch_size_per_gpu: 2

# Trainer configuration
trainer:
  total_training_steps: 200
  save_freq: 50
  test_freq: 10
  val_before_train: false
  critic_warmup: 0
  
  # Logging
  logger: ['console', 'wandb']
  project_name: "AgentFly-Training"
  experiment_name: "my_experiment"
  
  # Hardware
  n_gpus_per_node: 8
  nnodes: 1
```

### Custom Configuration Class

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml

@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Agent configuration
    agent_type: str = "react"
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    template: str = "qwen2.5-no-tool"
    tools: List[str] = field(default_factory=lambda: ["code_interpreter"])
    reward_name: str = "math_reward"
    max_steps: int = 8
    num_chains: int = 8
    
    # Data configuration
    train_files: str = "./data/train.json"
    val_files: str = "./data/val.json"
    train_batch_size: int = 64
    
    # Training configuration
    algorithm: str = "grpo"
    learning_rate: float = 5e-7
    total_training_steps: int = 200
    kl_coef: float = 0.001
    entropy_coeff: float = 0.001
    
    # Hardware configuration
    n_gpus_per_node: int = 8
    nnodes: int = 1
    response_length: int = 512
    
    # Monitoring
    project_name: str = "AgentFly-Training"
    experiment_name: str = "default_experiment"
    save_freq: int = 50
    test_freq: int = 10
    
    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested configuration
        flat_config = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_config[f"{key}_{subkey}"] = subvalue
            else:
                flat_config[key] = value
        
        return cls(**flat_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for command line arguments."""
        return {
            "agent.agent_type": self.agent_type,
            "agent.model_name_or_path": self.model_name_or_path,
            "agent.template": self.template,
            "agent.tools": str(self.tools),
            "agent.reward_name": self.reward_name,
            "agent.max_steps": self.max_steps,
            "agent.num_chains": self.num_chains,
            "data.train_files": self.train_files,
            "data.val_files": self.val_files,
            "data.train_batch_size": self.train_batch_size,
            "algorithm.adv_estimator": self.algorithm,
            "actor_rollout_ref.actor.optim.lr": self.learning_rate,
            "trainer.total_training_steps": self.total_training_steps,
            "trainer.project_name": self.project_name,
            "trainer.experiment_name": self.experiment_name,
            "trainer.n_gpus_per_node": self.n_gpus_per_node,
            "trainer.nnodes": self.nnodes,
        }
    
    def generate_experiment_name(self) -> str:
        """Generate descriptive experiment name."""
        model_short = self.model_name_or_path.split('/')[-1]
        return f"{model_short}-{self.agent_type}-{self.algorithm}-lr{self.learning_rate}-bs{self.train_batch_size}-kl{self.kl_coef}"

# Usage
config = TrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    train_files="./data/math_train.json",
    experiment_name="math_problem_solving"
)

config.experiment_name = config.generate_experiment_name()
```

## Training Scripts

### Basic Training Script

Create `scripts/train_basic.sh`:

```bash
#!/bin/bash

# Basic training script for AgentFly

set -e  # Exit on error

# Configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
TEMPLATE="qwen2.5-no-tool"
AGENT_TYPE="react"
TOOLS='["code_interpreter", "answer"]'
REWARD="math_reward"

# Training parameters
LR=5e-7
BATCH_SIZE=64
NUM_CHAINS=8
TOTAL_STEPS=200
KL_COEF=0.001
ENTROPY_COEFF=0.001

# Data paths
TRAIN_DATA="./data/train.json"
VAL_DATA="./data/val.json"

# Experiment settings
PROJECT_NAME="AgentFly-Tutorial"
EXPERIMENT_NAME="${MODEL##*/}-${AGENT_TYPE}-${REWARD}-$(date +%Y%m%d_%H%M%S)"

echo "Starting training: $EXPERIMENT_NAME"

# Start training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$BATCH_SIZE \
    agent.agent_type=$AGENT_TYPE \
    agent.tools="$TOOLS" \
    agent.template=$TEMPLATE \
    agent.model_name_or_path=$MODEL \
    agent.max_steps=8 \
    agent.backend=async_verl \
    agent.reward_name=$REWARD \
    agent.num_chains=$NUM_CHAINS \
    agent.use_agent=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=$MODEL \
    critic.ppo_mini_batch_size=$BATCH_SIZE \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=0 \
    trainer.logger='[console,wandb]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.val_before_train=False

echo "Training completed: $EXPERIMENT_NAME"
```

### Advanced Training Script with Configuration

Create `scripts/train_advanced.py`:

```python
#!/usr/bin/env python3
"""
Advanced training script with configuration management.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingManager:
    """Manage AgentFly training with advanced features."""
    
    def __init__(self, config_file: str):
        """Initialize training manager."""
        self.config_file = Path(config_file)
        self.config = self.load_config()
        self.setup_environment()
    
    def load_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        self.validate_config(config)
        
        return config
    
    def validate_config(self, config: Dict[str, Any]):
        """Validate training configuration."""
        required_sections = ['agent', 'data', 'training', 'hardware']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate data files exist
        train_files = config['data']['train_files']
        if isinstance(train_files, str):
            train_files = [train_files]
        
        for file_path in train_files:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Training file not found: {file_path}")
        
        logger.info("Configuration validation passed")
    
    def setup_environment(self):
        """Setup training environment."""
        # Set environment variables
        os.environ['VLLM_USE_V1'] = '1'
        os.environ['HYDRA_FULL_ERROR'] = '1'
        
        # Setup Ray if needed
        if self.config['hardware'].get('use_ray', True):
            self.setup_ray_cluster()
    
    def setup_ray_cluster(self):
        """Setup Ray cluster for distributed training."""
        logger.info("Setting up Ray cluster...")
        
        # Stop existing Ray cluster
        subprocess.run(['ray', 'stop'], check=False)
        subprocess.run(['rm', '-rf', '/tmp/ray/ray_current_cluster'], check=False)
        
        # Get cluster configuration
        hardware_config = self.config['hardware']
        head_node_ip = subprocess.check_output(['hostname', '--ip-address']).decode().strip()
        port = hardware_config.get('ray_port', 6379)
        num_cpus = hardware_config.get('num_cpus', 192)
        num_gpus = hardware_config.get('num_gpus', 8)
        
        # Start Ray head node
        ray_cmd = [
            'ray', 'start', '--head',
            f'--node-ip-address={head_node_ip}',
            f'--port={port}',
            f'--num-cpus={num_cpus}',
            f'--num-gpus={num_gpus}'
        ]
        
        result = subprocess.run(ray_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to start Ray: {result.stderr}")
            raise RuntimeError("Ray cluster setup failed")
        
        logger.info(f"Ray cluster started on {head_node_ip}:{port}")
    
    def build_training_command(self) -> List[str]:
        """Build the training command."""
        config = self.config
        
        # Base command
        cmd = ['python3', '-m', 'verl.trainer.main_ppo']
        
        # Add algorithm configuration
        training_config = config['training']
        cmd.extend([
            f"algorithm.adv_estimator={training_config['algorithm']}",
            f"algorithm.kl_ctrl.kl_coef={training_config['kl_coef']}"
        ])
        
        # Add data configuration
        data_config = config['data']
        cmd.extend([
            f"data.train_files={data_config['train_files']}",
            f"data.val_files={data_config['val_files']}",
            f"data.train_batch_size={data_config['train_batch_size']}"
        ])
        
        # Add agent configuration
        agent_config = config['agent']
        tools_str = str(agent_config['tools']).replace("'", '"')
        cmd.extend([
            f"agent.agent_type={agent_config['agent_type']}",
            f"agent.model_name_or_path={agent_config['model_name_or_path']}",
            f"agent.template={agent_config['template']}",
            f"agent.tools={tools_str}",
            f"agent.reward_name={agent_config['reward_name']}",
            f"agent.max_steps={agent_config['max_steps']}",
            f"agent.num_chains={agent_config['num_chains']}",
            f"agent.backend={agent_config.get('backend', 'async_verl')}",
            "agent.use_agent=True"
        ])
        
        # Add model and training parameters
        model_path = agent_config['model_name_or_path']
        cmd.extend([
            f"actor_rollout_ref.actor.optim.lr={training_config['learning_rate']}",
            f"actor_rollout_ref.model.path={model_path}",
            "actor_rollout_ref.model.use_remove_padding=False",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={data_config['train_batch_size']}",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
            "actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef={training_config['kl_coef']}",
            "actor_rollout_ref.actor.kl_loss_type=mse",
            f"actor_rollout_ref.actor.entropy_coeff={training_config['entropy_coeff']}",
            "actor_rollout_ref.model.enable_gradient_checkpointing=False",
            "actor_rollout_ref.actor.fsdp_config.param_offload=True",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={config['hardware'].get('tensor_parallel_size', 2)}",
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.response_length={training_config.get('response_length', 512)}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={config['hardware'].get('gpu_memory_utilization', 0.5)}",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            f"critic.model.path={model_path}",
            f"critic.ppo_mini_batch_size={data_config['train_batch_size']}",
            "critic.ppo_micro_batch_size_per_gpu=2"
        ])
        
        # Add trainer configuration
        monitoring_config = config.get('monitoring', {})
        cmd.extend([
            f"trainer.total_training_steps={training_config['total_training_steps']}",
            f"trainer.save_freq={training_config.get('save_freq', 50)}",
            f"trainer.test_freq={training_config.get('test_freq', 10)}",
            "trainer.val_before_train=False",
            "trainer.critic_warmup=0",
            f"trainer.logger={monitoring_config.get('loggers', ['console', 'wandb'])}",
            f"trainer.project_name={monitoring_config.get('project_name', 'AgentFly')}",
            f"trainer.experiment_name={monitoring_config.get('experiment_name', 'default')}",
            f"trainer.n_gpus_per_node={config['hardware']['num_gpus']}",
            f"trainer.nnodes={config['hardware'].get('num_nodes', 1)}"
        ])
        
        return cmd
    
    def run_training(self, dry_run: bool = False):
        """Run the training process."""
        cmd = self.build_training_command()
        
        if dry_run:
            logger.info("Dry run - would execute:")
            logger.info(" ".join(cmd))
            return
        
        logger.info("Starting training...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Create output directory
        output_dir = Path("outputs") / self.config['monitoring']['experiment_name']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration for reproducibility
        config_backup = output_dir / "training_config.yaml"
        with open(config_backup, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Run training
        try:
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Log output
            log_file = output_dir / "training.log"
            with open(log_file, 'w') as f:
                f.write(result.stdout)
            
            if result.returncode == 0:
                logger.info("Training completed successfully")
            else:
                logger.error(f"Training failed with return code {result.returncode}")
                logger.error(result.stdout[-1000:])  # Last 1000 chars
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with exception: {str(e)}")
            raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="AgentFly Training Manager")
    parser.add_argument("config", help="Path to training configuration YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    
    args = parser.parse_args()
    
    try:
        manager = TrainingManager(args.config)
        
        if args.validate_only:
            logger.info("Configuration is valid")
            return
        
        manager.run_training(dry_run=args.dry_run)
        
    except Exception as e:
        logger.error(f"Training setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Quick Start Script

Create `scripts/quick_start.py`:

```python
#!/usr/bin/env python3
"""
Quick start script for AgentFly training.
"""

import json
import subprocess
from pathlib import Path

def create_sample_data():
    """Create sample training data."""
    
    # Simple math problems
    train_data = [
        {
            "question": "What is 15 + 27?",
            "answer": "42",
            "id": "math_001",
            "difficulty": "easy"
        },
        {
            "question": "Calculate 8 × 7",
            "answer": "56", 
            "id": "math_002",
            "difficulty": "easy"
        },
        {
            "question": "What is 144 ÷ 12?",
            "answer": "12",
            "id": "math_003",
            "difficulty": "medium"
        }
    ] * 50  # Repeat to have enough data
    
    val_data = train_data[:20]  # Use subset for validation
    
    # Save data
    Path("data").mkdir(exist_ok=True)
    
    with open("data/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open("data/val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Created {len(train_data)} training samples and {len(val_data)} validation samples")

def run_quick_training():
    """Run a quick training session."""
    
    # Create sample data
    create_sample_data()
    
    # Quick training command
    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "data.train_files=./data/train.json",
        "data.val_files=./data/val.json", 
        "data.train_batch_size=16",
        "agent.agent_type=react",
        "agent.tools=[answer]",
        "agent.template=qwen2.5-no-tool",
        "agent.model_name_or_path=Qwen/Qwen2.5-3B-Instruct",
        "agent.max_steps=3",
        "agent.num_chains=4",
        "agent.reward_name=exact_match_reward",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.rollout.response_length=256",
        "trainer.total_training_steps=10",
        "trainer.save_freq=5",
        "trainer.test_freq=5",
        "trainer.project_name=QuickStart",
        "trainer.experiment_name=quick_test"
    ]
    
    print("Starting quick training session...")
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print("Quick training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
    except KeyboardInterrupt:
        print("Training interrupted by user")

if __name__ == "__main__":
    run_quick_training()
```

## Distributed Training

### Multi-Node Training Script

Create `scripts/train_multinode.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=agentfly-training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.log

# Multi-node training script for SLURM

set -e

# Configuration
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

# Get node information
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Training parameters
MODEL="Qwen/Qwen2.5-7B-Instruct"
TOTAL_GPUS=$((SLURM_NNODES * 8))
BATCH_SIZE_PER_GPU=8
GLOBAL_BATCH_SIZE=$((TOTAL_GPUS * BATCH_SIZE_PER_GPU))

echo "Training on $SLURM_NNODES nodes with $TOTAL_GPUS total GPUs"
echo "Head node: $head_node ($head_node_ip)"

# Start Ray cluster
echo "Starting Ray cluster..."
port=6379
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus=64 --num-gpus=8 &

sleep 10

# Start worker nodes
for ((i=1; i<SLURM_NNODES; i++)); do
    node=${nodes_array[$i]}
    echo "Starting Ray worker on $node"
    srun --nodes=1 --ntasks=1 -w "$node" \
        ray start --address="$head_node_ip:$port" --num-cpus=64 --num-gpus=8 &
done

sleep 30

# Run training on head node
srun --nodes=1 --ntasks=1 -w "$head_node" python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/large_train.json \
    data.val_files=./data/large_val.json \
    data.train_batch_size=$GLOBAL_BATCH_SIZE \
    agent.agent_type=react \
    agent.model_name_or_path=$MODEL \
    agent.num_chains=16 \
    trainer.total_training_steps=1000 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$SLURM_NNODES \
    trainer.experiment_name="multinode_${SLURM_JOB_ID}"

echo "Training completed"
```

### Ray Cluster Management

Create `scripts/ray_manager.py`:

```python
#!/usr/bin/env python3
"""
Ray cluster management for AgentFly training.
"""

import subprocess
import time
import argparse
from typing import List, Dict
import yaml

class RayClusterManager:
    """Manage Ray clusters for distributed training."""
    
    def __init__(self, config_file: str = None):
        """Initialize Ray cluster manager."""
        self.config = self.load_config(config_file) if config_file else {}
        self.head_node_ip = None
        self.port = self.config.get('ray_port', 6379)
    
    def load_config(self, config_file: str) -> Dict:
        """Load cluster configuration."""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def start_head_node(self, num_cpus: int = 192, num_gpus: int = 8):
        """Start Ray head node."""
        print("Starting Ray head node...")
        
        # Get head node IP
        result = subprocess.run(['hostname', '--ip-address'], 
                              capture_output=True, text=True)
        self.head_node_ip = result.stdout.strip()
        
        # Start head node
        cmd = [
            'ray', 'start', '--head',
            f'--node-ip-address={self.head_node_ip}',
            f'--port={self.port}',
            f'--num-cpus={num_cpus}',
            f'--num-gpus={num_gpus}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start head node: {result.stderr}")
        
        print(f"Head node started at {self.head_node_ip}:{self.port}")
        return self.head_node_ip
    
    def add_worker_node(self, worker_ip: str, num_cpus: int = 192, num_gpus: int = 8):
        """Add worker node to cluster."""
        if not self.head_node_ip:
            raise RuntimeError("Head node not started")
        
        print(f"Adding worker node {worker_ip}...")
        
        cmd = [
            'ssh', worker_ip,
            'ray', 'start',
            f'--address={self.head_node_ip}:{self.port}',
            f'--num-cpus={num_cpus}',
            f'--num-gpus={num_gpus}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Failed to add worker {worker_ip}: {result.stderr}")
        else:
            print(f"Worker node {worker_ip} added successfully")
    
    def setup_cluster(self, worker_ips: List[str]):
        """Setup complete Ray cluster."""
        # Stop any existing cluster
        self.stop_cluster()
        
        # Start head node
        self.start_head_node()
        
        # Add worker nodes
        time.sleep(5)  # Wait for head node to be ready
        for worker_ip in worker_ips:
            self.add_worker_node(worker_ip)
        
        # Wait for cluster to stabilize
        time.sleep(10)
        
        # Verify cluster
        self.verify_cluster()
    
    def stop_cluster(self):
        """Stop Ray cluster."""
        print("Stopping Ray cluster...")
        subprocess.run(['ray', 'stop'], check=False)
        subprocess.run(['rm', '-rf', '/tmp/ray/ray_current_cluster'], check=False)
    
    def verify_cluster(self):
        """Verify cluster status."""
        try:
            result = subprocess.run(['ray', 'status'], 
                                  capture_output=True, text=True, timeout=30)
            print("Cluster status:")
            print(result.stdout)
        except subprocess.TimeoutExpired:
            print("Warning: Cluster status check timed out")
        except Exception as e:
            print(f"Warning: Could not verify cluster status: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Ray Cluster Manager")
    parser.add_argument("--config", help="Cluster configuration file")
    parser.add_argument("--workers", nargs="+", help="Worker node IPs")
    parser.add_argument("--stop", action="store_true", help="Stop cluster")
    parser.add_argument("--status", action="store_true", help="Show cluster status")
    
    args = parser.parse_args()
    
    manager = RayClusterManager(args.config)
    
    if args.stop:
        manager.stop_cluster()
    elif args.status:
        manager.verify_cluster()
    elif args.workers:
        manager.setup_cluster(args.workers)
    else:
        manager.start_head_node()

if __name__ == "__main__":
    main()
```

## Monitoring and Logging

### Weights & Biases Integration

```python
import wandb
from typing import Dict, Any

class TrainingMonitor:
    """Monitor training progress with W&B and local logging."""
    
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any]):
        """Initialize monitoring."""
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        # Initialize W&B
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config
        )
        
        # Setup local logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/{experiment_name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics."""
        # Log to W&B
        wandb.log(metrics, step=step)
        
        # Log locally
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {metrics_str}")
    
    def log_artifacts(self, file_paths: List[str]):
        """Log artifacts to W&B."""
        for file_path in file_paths:
            wandb.save(file_path)
    
    def finish(self):
        """Finish monitoring."""
        wandb.finish()
```

### Custom Metrics Tracking

```python
class MetricsTracker:
    """Track custom training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history = []
        self.current_metrics = {}
    
    def add_metric(self, name: str, value: float, step: int):
        """Add a metric value."""
        self.current_metrics[name] = value
        
        # Store in history
        self.metrics_history.append({
            'step': step,
            'name': name,
            'value': value,
            'timestamp': time.time()
        })
    
    def compute_moving_average(self, metric_name: str, window: int = 10) -> float:
        """Compute moving average of a metric."""
        recent_values = [
            m['value'] for m in self.metrics_history[-window:]
            if m['name'] == metric_name
        ]
        
        return sum(recent_values) / len(recent_values) if recent_values else 0.0
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        values = [m['value'] for m in self.metrics_history if m['name'] == metric_name]
        
        if not values:
            return {}
        
        return {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1],
            'count': len(values)
        }
```

## Troubleshooting

### Common Issues and Solutions

```python
def troubleshooting_guide():
    """Common training issues and solutions."""
    
    issues = {
        "CUDA out of memory": {
            "symptoms": ["RuntimeError: CUDA out of memory", "GPU memory allocation failed"],
            "solutions": [
                "Reduce batch_size or micro_batch_size_per_gpu",
                "Enable gradient_checkpointing",
                "Reduce response_length",
                "Use model sharding (tensor_model_parallel_size > 1)"
            ]
        },
        
        "Ray cluster connection failed": {
            "symptoms": ["ConnectionError", "Ray cluster not accessible"],
            "solutions": [
                "Check Ray cluster status with 'ray status'",
                "Verify head node IP and port",
                "Restart Ray cluster",
                "Check firewall settings"
            ]
        },
        
        "Tool execution timeout": {
            "symptoms": ["Tool call timeout", "Environment allocation failed"],
            "solutions": [
                "Increase tool timeout settings",
                "Check environment pool_size",
                "Verify tool dependencies are installed",
                "Monitor resource usage"
            ]
        },
        
        "Training divergence": {
            "symptoms": ["Loss suddenly increases", "NaN in gradients"],
            "solutions": [
                "Reduce learning rate",
                "Increase KL penalty coefficient",
                "Check data quality",
                "Add gradient clipping"
            ]
        },
        
        "Slow training speed": {
            "symptoms": ["Low GPU utilization", "Long step times"],
            "solutions": [
                "Increase batch size",
                "Optimize data loading",
                "Check I/O bottlenecks",
                "Use faster storage for data"
            ]
        }
    }
    
    return issues

def diagnose_training_issue(log_file: str) -> List[str]:
    """Diagnose training issues from log file."""
    issues_found = []
    
    with open(log_file, 'r') as f:
        log_content = f.read().lower()
    
    # Check for common error patterns
    error_patterns = {
        "cuda out of memory": "CUDA memory issue",
        "connection refused": "Network/Ray issue",
        "timeout": "Timeout issue",
        "nan": "Numerical instability",
        "permission denied": "Permission issue"
    }
    
    for pattern, issue_type in error_patterns.items():
        if pattern in log_content:
            issues_found.append(issue_type)
    
    return issues_found
```

### Health Check Script

Create `scripts/health_check.py`:

```python
#!/usr/bin/env python3
"""
Health check script for AgentFly training environment.
"""

import subprocess
import sys
from pathlib import Path
import importlib

def check_python_packages():
    """Check required Python packages."""
    required_packages = [
        'torch', 'transformers', 'datasets', 'numpy', 'pandas',
        'wandb', 'ray', 'pydantic', 'hydra-core'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
    
    return len(missing_packages) == 0

def check_gpu_availability():
    """Check GPU availability and CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def check_ray_cluster():
    """Check Ray cluster status."""
    try:
        result = subprocess.run(['ray', 'status'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ray cluster running")
            return True
        else:
            print("❌ Ray cluster not running")
            return False
    except Exception as e:
        print(f"❌ Ray check failed: {e}")
        return False

def check_data_files(data_dir: str = "data"):
    """Check training data files."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    required_files = ["train.json", "val.json"]
    missing_files = []
    
    for file_name in required_files:
        file_path = data_path / file_name
        if file_path.exists():
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"✅ {file_name} ({file_size:.1f} KB)")
        else:
            print(f"❌ {file_name}")
            missing_files.append(file_name)
    
    return len(missing_files) == 0

def check_disk_space():
    """Check available disk space."""
    try:
        result = subprocess.run(['df', '-h', '.'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                fields = lines[1].split()
                available = fields[3]
                usage = fields[4]
                print(f"✅ Disk space: {available} available ({usage} used)")
                return True
    except Exception as e:
        print(f"❌ Disk space check failed: {e}")
    
    return False

def main():
    """Run all health checks."""
    print("AgentFly Training Environment Health Check")
    print("=" * 50)
    
    checks = [
        ("Python packages", check_python_packages),
        ("GPU availability", check_gpu_availability),
        ("Ray cluster", check_ray_cluster),
        ("Data files", check_data_files),
        ("Disk space", check_disk_space)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            passed = check_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All checks passed - ready for training!")
        sys.exit(0)
    else:
        print("❌ Some checks failed - please fix issues before training")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Next Steps

Now that you understand training setup, proceed to:
- [Tutorial 7: Complete Pipeline](07_complete_pipeline.md) for the end-to-end workflow

You now have all the knowledge needed to train your own customized AgentFly agents! The complete pipeline tutorial will show you how to put everything together into a working system.