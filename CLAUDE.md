# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation and Setup
```bash
# Install AgentFly and dependencies
pip install -e .
pip install -e '.[verl]' --no-build-isolation

# Initialize git submodules (required for VERL)
git submodule init
git submodule update

# Optional: Install Redis for distributed training
conda install conda-forge::redis-server==7.4.0
```

### Testing
```bash
# Run unit tests
pytest agents/tests/unit/

# Run tests with async support
python -m pytest --asyncio-mode=auto

# Run specific test file
pytest agents/tests/unit/test_agents.py

# Run tests with coverage
pytest --cov=agents agents/tests/
```

### Training
```bash
# Run RL training for code interpreter agent
cd verl
bash examples/run_agents/run_code_agent.sh

# Run React agent training
bash examples/run_agents/run_react_agent.sh
```

### Code Quality
```bash
# Format code with Ruff
ruff format .

# Check linting
ruff check .

# Fix linting issues automatically
ruff check --fix .
```

## Architecture Overview

### Core Components

**Agent Framework** (`agents/agents/agents/`)
- `BaseAgent`: Abstract base class defining the agent interface
- `ReactAgent`: Implements ReAct reasoning pattern with tool calling
- `CodeAgent`: Specialized for code interpretation tasks
- `ThinkAgent`: Chain-of-thought reasoning without tools
- Templates in `templates/` provide multi-modal prompt formats for different LLMs

**Tool System** (`agents/agents/tools/`)
- Decorator-based tool definition using `@tool`
- Supports async execution for high throughput
- Environment-specific tools for AlfWorld, WebShop, ScienceWorld
- Code interpreter with isolated Docker execution

**Environment Management** (`agents/agents/envs/`)
- Docker-based isolation for code execution
- Resource pool with warm containers for efficiency
- Centralized environment coordination via Redis
- Support for multiple specialized environments

**Reward Functions** (`agents/agents/rewards/`)
- Decorator-based reward definition using `@reward`
- Task-specific rewards for math, QA, code tasks
- LLM-as-judge reward system for complex evaluations
- Async execution for parallel reward computation

**VERL Integration** (`verl/`)
- Reinforcement learning training framework
- Supports GRPO, Reinforce++ algorithms
- Token-level masking for multi-turn training
- Distributed training with Ray

### Key Design Patterns

1. **Decorator Pattern**: Tools and rewards are defined using simple decorators, making extension easy
2. **Factory Pattern**: Agent creation through factory methods with configuration
3. **Pool Pattern**: Resource management using warm container pools
4. **Async/Await**: Extensive use of async for parallel tool and reward execution

### Configuration

- Agent configs: YAML files in `agents/agents/configs/`
- Training configs: Hydra-based configs in VERL
- Environment configs: Docker and resource settings per environment

### Development Workflow

0. see @ROADMAP.md for current status and next steps.
1. When adding new tools: Create in `agents/agents/tools/` with `@tool` decorator
2. When adding new rewards: Create in `agents/agents/rewards/` with `@reward` decorator
3. When modifying agents: Follow the `BaseAgent` interface in `agents/agents/agents/base.py`
4. When adding tests: Place in appropriate subdirectory under `agents/tests/unit/`
5. Always run tests and linting before committing changes
6. You can just modify ui/gui files, you dont have the permission to modify others.