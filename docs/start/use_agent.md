## Build an Agent

### Use a Predefined Agent
We can specify the following arguments to use a predefined agent:

- model_name: the path or name or the model, used to load weights
- tools: tools that will be used by the agent
- template: chat template
- backend: what type of backend

The following shows an example to use Qwen2.5-7B-Instruct as a react agent:

```python
from agents.agents.react.react_agent import ReactAgent
from agents.tools.src.code.tools import code_interpreter
from agents.tools.src.search.google_search import google_search_serper
from agents.tools.src.react.tools import answer

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


