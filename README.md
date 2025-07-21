# AgentFly: Scalable and Extendable Reinforcement Larning for LLM Agents

This library is an extandable framework for building LLM agents with reinforcement learning. It provides a flexible and powerful system for creating agent that interact with tools, learn from rewards in multi-turn manner and complete tasks automatically.
![Overview](assets/images/overview.png)

## Installation
```bash
git submodule init
git submodule update
pip install -r agents/requirements.txt
pip install -r verl/requirements.txt
```
Optional
We need redis-server for caching search results
```bash
conda install conda-forge::redis-server==7.4.0
```
Some of our environments are managed by *enroot* backend. To use them, please install [enroot](https://github.com/NVIDIA/enroot/blob/master/doc/installation.md).

## Run Example Training
Suppose you are in a compute node (with 8 gpus).

Run RL training of code
```python
cd verl
bash examples/run_agents/run_code_agent.sh
```


## Features
### 1. Multi-Chain Agent Rollout and Multi-Turn Training
To support algorithms like GRPO, Reinforce++, we design multi-chain inference, enabling agents to solve one task with multiple paths at the same time. We build RL computation and update LLMs in multi-turn manner by applying token masks. The training is based on [verl](https://github.com/volcengine/verl).


### 2. Simple Tool and Reward Integration
Define tools and rewards, which can be used directly by agents.
```python
@tool(name=...)
def customized_tool(...):
    ...

def custmozed_reward(...):
    ...

agent = ReactAgent(
    model_name,
    tools=[customized_tool],
    reward=customized_reward
)
```

### 3. Easy Development
Decoupled agent and training module. Simply customize your own agent, which can directly be applied to training.


## Contribute & Discussion
[WeChat|微信](assets/images/wechat.jpg)

[Discord](https://discord.gg/CchUj7Sp)

## Training Curves
Reward curves on Qwen2.5-Instruct 3B and 7B models.
![Curves](assets/images/training_curves.png)
