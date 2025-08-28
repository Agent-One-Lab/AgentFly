# Define Tools & Reward Functions

We have shown how to build an agent, to further customize the training, we need to define tools and reward functions.

**Tool Definition**

Define a tools is simple and easy in AgentFly framework. You simple write a function, and then decorate it with `@tool`. The following example shows the calculator tool we used previously.

```python
from agentfly.tools import tool
from sympy import simplify, sympify, Rational

@tool(name="calculator", description="Calculate the result of a mathematical expression.")
def calculate(expression: str):
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

@reward(name="math_reward_string_equal")
def math_reward_string_equal(prediction: str, answer: str) -> float:
    import re

    def extract_last_number(s: str):
        matches = re.findall(r'\d+', s)  # find all sequences of digits
        return int(matches[-1]) if matches else None

    prediction = extract_last_number(prediction)
    
    if prediction == answer:
        return 1.0
    else:
        return 0.0
```

Now we can use the agent with the reward function we just defined.

```python
from agentfly.agents import HFAgent
from agentfly.tools import calculate
agent = HFAgent(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    tools=[calculate],
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
    num_chains=1
)
```

Now we can get the trajectories and rewards with following code:
```python
trajectories = agent.trajectories
rewards = agent.rewards
print(trajectories)
print(rewards)
```

