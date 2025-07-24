# AgentFly: Scalable and Extensible Reinforcement Learning for LLM Agents

<p align="center">
<a href="https://arxiv.org/pdf/2507.14897" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Paper-arXiv-%23ffc8dd?style=plastic&link=https%3A%2F%2Farxiv.org%2Fpdf%2F2507.14897"></a>
<a href="https://agentfly.readthedocs.io/" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Docs-AgentFly-%23a2d2ff?style=plastic&link=https%3A%2F%2Fagentfly.readthedocs.io%2F"></a>
<a href="https://agentfly.readthedocs.io/" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Model-%F0%9F%A4%97HF-%23ffb703"></a>
</p>


This library is an extensible framework for building LLM agents with reinforcement learning. It provides a flexible and powerful system for creating agent that interact with tools, learn from rewards in multi-turn manner and complete tasks automatically.
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

## Training
### Run Example Training
Suppose you are in a compute node (with 8 gpus). We have prepared some training scripts for different tasks and tools in `verl/examples/run_agents/`. The script will try to download our prepared datasets and run training.

Run RL training of code_interpreter.
```python
cd verl
bash examples/run_agents/run_code_agent.sh
```
### Customized Training
To customize your own training, you need to prepare: 1. Datasets. 2. Define or use existing tools. 3. Define or use existing rewards. 3. Define your own agents.

#### 1. Data Format:
Data should be a json file, which contain a list of dicts with the following keys:
```json
[
    {
        "question": ...
        "optional_field1": ...
        "optional_field2": ...
        ...
    }
]
```
During training, `question` will be used to format the input messages, while other fields can be used in reward function.
```json
{
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": question}]}
    ]
    "optional_field1": ...
    "optional_field2": ...
    ...
}
```
#### 2. Tools & Rewards
You can use any existing tool, which is in [documentation](https://agentfly.readthedocs.io/), or define a tool by decorating it with `@tool`.
```python
@reward(name="customized_tool")
def customized_reward(arg1, arg2):
    # tool logic here
```

Define your reward function or use an existing one. The reward function can accept `prediction` and `trajectory` as the argument, which is the agent's final response and the whole trajectory. Other fields will also be given if you defined them in dataset. To use them, simply put these fields as arguments in reward function.

```python
@reward(name="customized_reward")
def customized_reward(prediction, trajectory, optional_field1, optional_field2):
    # calculate reward
    ...
```
#### 3. Agents


## Demo
1. The following shows an example of WebShop agent.

2. What does the training look like. During training, the resource system will dynamically allocate environments.

3. Monitoring training on [WANDB](https://wandb.ai/). Items include number of turns for each step, numer of tool calls, allocated environments.

https://github.com/user-attachments/assets/b8f42534-8d40-48a0-a264-f378e479bb3a



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
