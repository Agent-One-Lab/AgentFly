# Tool System

## Unifying Interfaces with Tool Call

Tool calling are becoming stardards for LM agents. We use *tool* as an abstractive concept that represent all *functions*, *tools*, *APIs*, *environments*, etc. Therefore, agent rollout is unified as a repeatitive process of generation and tool calling. However, there are some challenges that we need to tackle for RL rollout:

1. Parallelism: To ensure efficiency, during the rollout we need to have multiple interactions in parallel. This requires the tool can be called in parallel. In AgentFly (and many other frameworks), this is achieved through *asynchronous pipeline*: All of **generation, tool calling, and reward calculation are designed to be asynchronous**.

2. Isolation: Some environments need to be isolated during the interaction (e.g. writing files in an os). We mainly achieve this through running **multiple docker containers**.

### Definition

We have designed three ways to define tools:

1. Using `@tool` to define the tool
```python
from agentfly.tools import tool

@tool(name="tool name", description="Description about the tool)
async def custom_tool(arg1, arg2):
    """
    Description about the tool.

    Args:
        arg1: arg1 of the tool
        arg2: arg2 of the tool

    Returns:
        An observation string.
    """
    # tool logic here
    observation = "This is the return"
    return observation
```

2. Inheriting from the `BaseTool` class. The tool execution interface is the call method of the class.
```python
from agentfly.tools import BaseTool
class APITool(BaseTool)
    name="api_tool"
    description="This is a tool that uses API key"

    def __init__(self, api_key):
        self.api_key = api_key

    def call(self, query):
        """
        This is a tool that uses API key

        Args:
            query (str): query to use for the tool

        Returns:
            observation: observation string
        """

        return f"Call with query {query} and key {self.api_key}"
```


3. A specialized tool that can be directly defined as the method of the agent. This is recommended for specialized agent.
```python
from agentfly.agents import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, special_api_key, **kwargs):
        self.special_api_key = special_api_key
        super().__init__(**kwargs)

    def processing_query(self, query):
        return query

    @tool(name="specialized_tool")
    async def specialized_tool(self, query: str):
        query = self.processing_query(query)
        result = f"Call query {query} with special api key: {self.special_api_key}"

        return result
```




## Non-Stateful & Stateful Tool
We define two types of tools:

1. Non-Stateful Tool does not keep environment states. For such tool, we can simply write a function and decorate it with `@tool`.

    ```python
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
    ```

2. Stateful Tool keeps environmental state across turns or actions. In the new design, **stateful tools acquire resources directly via `Context` and `ResourceSpec`** instead of specifying `env_cls` on the decorator. The resource engine manages pooling and reuse; tools just request the resource they need and call its `step`-like interface.

    ```python
    from agentfly.core import Context
    from agentfly.envs.python_env import PythonSandboxSpec
    from agentfly.tools import tool

    @tool(
        name="code_interpreter",
        description="Run code in a sandboxed Python container and return stdout/stderr",
        stateful=True,
    )
    async def code_interpreter(code: str, context: Context) -> str:
        """
        Run the code in a sandboxed container and return the output from stdout or stderr.

        Args:
            code (str): The code to run.
            context (Context): Injected rollout context; used to acquire the Python sandbox resource.

        Returns:
            str: The output from stdout or stderr.
        """
        code = str(code)
        env = await context.acquire_resource(spec=PythonSandboxSpec, scope="global", backend="local")
        try:
            obs = await env.step(code)
            return str(obs)
        except Exception as e:
            return f"Error: {str(e)}"
    ```

## Tool Calling

### Asynchronous
Tool call are defined to be asynchronous for efficiency. Use `async` to define the tool. Note that defining a tool to be asynchronous is not just about using `async` keyword. The actual operation inside the tool call should be asynchronous.

!!! note
    A tool can also be defined to be synchronous in AgentFly. Although it might be slow for training.

### Isolation

For stateful tools, **isolation and sharing are handled by `Context` and `ResourceEngine`**, rather than by passing an explicit `id` into the tool:

- `Context.acquire_resource(spec=..., scope="rollout" | "global", backend="local")` uses a stable resource id (by default the rollout id) so that multiple tool calls in the same rollout share the same resource instance when they use the same `spec` and `scope`.
- Different rollouts automatically get different resource instances; you do not need to manually manage ids or release handles inside the tool.
- The underlying `ResourceSpec` (e.g., `max_global_num`) and engine configuration control **pool size and scaling**, rather than `pool_size` on the decorator.

This design lets tools focus on logic (`env.step(action)`) while the engine manages isolation, pooling, and reuse behind the scenes.

## Tool Calling Formats

In practice, the **format** used for tool calls can significantly affect the stability of agentic RL:

- Many base models struggle to reliably emit deeply nested **JSON** when arguments contain code or long texts, which can break parsing mid-training.
- XML-style wrappers (e.g. `<search>...</search>`) have shown more stable behavior for search-style tools in our experiments (similar to the Search-R1 setting).

AgentFly’s tool system is **format-agnostic** at the framework level—the parsing is implemented in the agent / template layer (e.g. `HFAgent`, `SearchR1Agent`). When designing new agents and templates for RL:

- Prefer **simple, robust formats** (XML-style tags or shallow JSON) for tool calls.
- Make sure the parsing logic is tolerant to minor generation noise (truncation, extra text) so that rollouts remain usable throughout long training runs.
