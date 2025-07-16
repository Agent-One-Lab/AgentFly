# Agents: Extendable Language Agent Framework for Reinforcement Learning
## Introduction
### Overview
The Agents library is an extendable framework for building language agents with reinforcement learning capabilities. It provides a flexible and powerful system for creating agents that can interact with tools, generate responses, and learn from rewards in a multi-turn conversation setting.

### Key Features
- Multi-turn Interaction: Agents can engage in extended conversations, using tools and generating responses based on observations
- Multi-chain Generation: Support for generating multiple response chains for each input message
- Extensible Architecture: Easy to create custom agents, tools, and reward functions
- Service Management: Built-in support for deploying and managing services (Docker, Redis, etc.)
- Tool Integration: Built-in support for tool calling and execution
- Reward System: Flexible reward function implementation for reinforcement learning

### Installation
```
pip install -r requirements.txt
```

### Quick Start
Here's a simple example to get started with the Agents framework:

Before we start, deploy the backend service for tool calling:
```bash
python -m tools.services.docker
```
Then run the following code to load an agent, and solve the question automatically.

```python
from agents.agents.auto import AutoAgent

# Configure the agent
config = {
    "agent_type": "react",
    "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    "template": "qwen-7b-chat",
    "tools": ["code_interpreter"],
    "reward_name": "math_reward",
    "vllm": True
}

# Create an agent instance
agent = AutoAgent.from_config(config)

# Prepare input messages
messages = [
    {
        "messages": [
            {"role": "user", "content": "What is the value of 3**10?"}
        ],
        "question": "What is the value of 3**10?",
        "answer": "59049"
    }
]

# Run the agent
agent.run(
    max_steps=3,
    start_messages=messages,
    num_chains=5
)

# Get results
trajectories = agent.trajectories
rewards = agent.rewards

```

### Customed Agents
- **Customize Tools**
    - Customize a tool is easy: simply decorate the function with `@tool`, the name, description, and schema will be captured automatically, you can also pass them as parameters.
    ```python
    from agents.tools import tool
    @tool(name="AdditionTool", description="Adds two numbers.")
    def add(a, b: int = 1):
        """
        Adds two numbers.

        Args:
            a (int): The first number.
            b (int): The second number which should be a non-negative integer.
        
        Returns:
            int: The sum of a and b.
        """
        return a + b
    
    print(add.name, add.description, add.schema)
    ```
    
 - **Customize Reward Function**
    - Use `@reward()` to decorate a function to make it a reward function usable by agents
    ```python
    from agents.rewards import reward
    @reward(name="string_equal_reward")
    def StringEqualReward(prediction: str, golden_answer: str):
        if prediction.lower() == golden_answer:
            return 1.0
        else:
            return 0.0
    ``` 
    - Reward function much either return a float value representing the the reward, or return a dictionary containing `reward` as the key, and all other values must be float values.
    ```python
    return_values = 0.75
    return_values = {"reward": 0.75, "f1": 0.50, "em": 0.0, "recall": 0.44}
    ```

- **Customize Agent**

