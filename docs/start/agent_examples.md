## Build an Agent

### Use a Predefined Agent
We can specify the following arguments to use a predefined agent:

- model_name: the path or name or the model, used to load weights
- tools: tools that will be used by the agent
- template: chat template
- backend: what type of backend

The following shows an example to use Qwen2.5-7B-Instruct as a react agent:

```python
from agentfly.agents.react.react_agent import ReactAgent
from agentfly.tools.src.code.tools import code_interpreter
from agentfly.tools.src.search.google_search import google_search_serper
from agentfly.tools.src.react.tools import answer

tools = [google_search_serper, answer]

task_info = "Use code to get answers. Result must be printed."

react_agent = ReactAgent(
    "Qwen/Qwen2.5-7B-Instruct",
    tools=tools,
    template="qwen2.5-no-tool",
    task_info=task_info,
    backend="async_vllm"
)

question = "Solve the equation 2x + 5y = 4 such that sum of x and y is 7."
messages = [
    {
        "messages": [
            {"role": "user", "content": f"{question}"}
        ],
        "question": f"{question}",
    },
]

await react_agent.run_async(
    max_steps=4,
    start_messages=messages,
    num_chains=5 # for the question, the agent will generate 5 trajectories
)

```

After the rollout, we can obtain the trajectories:

```python
react_agent.trajectories
```

Obtaining the rewards (if you specified reward function and give necessary parameters in input messages)
```
react_agent.rewards)
```

### Customize Agent

You can customize your own agent by defining how the agent do generation and handle tool calls.

```python
class CustomizedAgent(BaseAgent):
    def __init__(self,
        **kwargs
    )
        super().__init__(**kwargs)

    async def generate_async(self, messages_list: List[List[Dict]], **args):
        return await self.llm_engine.generate_async(messages_list, **args)

    def parse(self, responses: List(str), tools):
        # parse responses into tool calls
        ...
```

### Use Trained Agent
We provide the following agent that we can try:

- WebShop Agent:
```python
import asyncio
from agentfly.agents import ReactAgent
from agentfly.tools import webshop_browser
from agentfly.rewards import webshop_reward
from agentfly.agents.chain.streaming_observer import ConsoleStreamObserver

tools = [webshop_browser]

agent = ReactAgent(
    "Agent-One/Qwen2.5-3B-Instruct-WebShop",
    tools=tools,
    template="qwen2.5",
    backend="async_vllm",
    streaming="console"
)

question = "I am looking for a gluten free, 100% vegan plant based protein shake that is soy-free, and price lower than 40.00 dollars"

messages = [
    {
        "messages": [
            {"role": "user", "content": f"{question}"}
        ],
        "question": f"{question}"
    },
]

await agent.run_async(
    max_steps=10,
    start_messages=messages,
    num_chains=1,
    enable_streaming=True
)
```
- Science World Agent
```python
from agentfly.tools import scienceworld_explorer
from agentfly.rewards import scienceworld_reward


tools = [scienceworld_explorer]

agent = ReactAgent(
    "Agent-One/Qwen2.5-7B-Instruct-ScienceWorld-REINFORCEPP",
    tools=tools,
    template="qwen2.5-no-system-tool",
    backend="async_vllm",
    reward_fn=scienceworld_reward,
    streaming="console"
)

# question = "Your task is to boil ice cream. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter."
# task_name = "boil"
# variation_idx = 14


# question = "Your task is to find a(n) animal. First, focus on the thing. Then, move it to the green box in the kitchen."
# task_name = "find-animal"
# variation_idx = 150

# question = "Your task is to find a(n) plant. First, focus on the thing. Then, move it to the purple box in the bathroom."
# task_name = "find-plant"
# variation_idx = 191


# question = "Your task is to grow a orange. This will require growing several plants, and them being crosspollinated to produce fruit.  Seeds can be found in the kitchen. To complete the task, focus on the grown orange."
# task_name = "grow-fruit"
# variation_idx = 92

# question = "Your task is to determine which of the two inclined planes (bronze, copper) has the most friction. After completing your experiment, focus on the inclined plane with the most friction."
# task_name = "inclined-plane-friction-named-surfaces"
# variation_idx = 776

question = "Your task is to measure the melting point of orange juice, which is located around the kitchen. First, focus on the thermometer. Next, focus on the orange juice. If the melting point of orange juice is above 200.0 degrees celsius, focus on the yellow box. If the melting point of orange juice is below 200.0 degrees celsius, focus on the purple box. The boxes are located around the kitchen."
task_name = "measure-melting-point-known-substance"
variation_idx = 247

messages = [
    {
        "messages": [
            {"role": "user", "content": f"{question}"}
        ],
        "question": f"{question}",
        "task_name": task_name,
        "variation_idx": variation_idx
    },
]

await agent.run_async(
    max_steps=20,
    start_messages=messages,
    num_chains=1,
    enable_streaming=True
)

print(agent.rewards)
```

- Retrieval Agent

```python
from agentfly.tools import dense_retrieve, asyncdense_retrieve

tools = [dense_retrieve]

agent = ReactAgent(
    "Agent-One/Qwen2.5-3B-Instruct-Retrieval-GRPO",
    tools=tools,
    template="qwen2.5-no-system-tool",
    backend="async_vllm",
    streaming="console"
)

question = "Who is Geoffrey Hinton"


messages = [
    {
        "messages": [
            {"role": "user", "content": f"{question}"}
        ],
        "question": f"{question}",
    },
]

await agent.run_async(
    max_steps=6,
    start_messages=messages,
    num_chains=1,
    enable_streaming=True
)
```