# Define Tools & Reward Functions

We have shown how to build an agent, to further customize the training, we need to define tools and reward functions.

**Tool Definition**

Define a tools is simple and easy in AgentFly framework. You simply write a function, and then decorate it with `@tool`. The following example shows the calculator tool we used previously. You can also inherit a `BaseTool` class to define the tool with more flexibility and complexity (refer to Features section).

```python
from agentfly.tools import tool
from sympy import simplify, sympify, Rational

@tool(name="calculator", description="Calculate the result of a mathematical expression.")
def calculator(expression: str):
    try:
        expr = sympify(expression)
        result = simplify(expr)

        # Check if the result is a number
        if result.is_number:
            # If the result is a rational number, return as a fraction
            if isinstance(result, Rational):
                return str(result)
            # If the result is a floating point number, format to remove redundant zeros
            else:
                return "{:g}".format(float(result))
        else:
            return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
```

Now we have the tool, we can then define the reward function, which also simply use a `@reward` decorator. The following example shows a reward by extracting the last number of a text and compare it with the golden asnwer. The return of the reward function is a float number representing the reward, or a dictionary containing "reward" as a key.

```python
from agentfly.rewards import reward
from typing import List, Dict
import re

@reward(name="math_reward_string_equal")
def math_reward_string_equal(prediction: str, answer: str, trajectory: List[Dict]) -> float:

    def extract_last_number(s: str):
        matches = re.findall(r'\d+', s)  # find all sequences of digits
        return matches[-1] if matches else None

    tool_count = 0
    for msg in trajectory:
        if msg["role"] == "tool":
            tool_count += 1
    
    if tool_count < 1:
        return 0.0
    else:
        prediction = extract_last_number(prediction)
        
        if prediction == answer:
            return 1.0
        else:
            return 0.1
```
Note that in this reward function, we use the trajectory to count how many tools the agent has called. If the agent called at least one, we give it the basic format reward (0.1), then if it further gets the answer correct, it gets the full reward (1.0).
Now we can use the agent with the reward function we just defined.

```python
from agentfly.agents import HFAgent
from agentfly.tools import calculator
agent = HFAgent(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    tools=[calculator],
    template="qwen2.5",
    reward_fn=math_reward_string_equal,
    backend="async_vllm",
)
```

Then we can run the agent and get rewards:

```python
messages = {
    "messages": [
        {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"}
    ],
    "answer": "72"
}
await agent.run(
    messages=messages,
    max_turns=3,
    num_chains=5 # Generate 5 trajectories for the query
)
```

Now we can get the trajectories and rewards with following code:
```python
trajectories = agent.trajectories
rewards = agent.rewards
print(trajectories)
print(rewards)
```

