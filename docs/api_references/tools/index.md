# Tools

AgentFly provides a comprehensive tool system that enables agents to interact with external systems and APIs. Tools can be stateful (with environment management) or stateless, and support both synchronous and asynchronous execution.

## Structure

- [Tool](tool.md) - Base tool class and decorator
- [Predefined Tools](predefined_tools.md) - Built-in tools

## Basic Tool Definition

```python
from agentfly.tools import tool

@tool(name="calculator", description="Calculate mathematical expressions")
def calculator(expression: str):
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
```

## Stateful Tool with ResourceEngine

```python
from agentfly.core import Context
from agentfly.envs.python_env import PythonSandboxSpec
from agentfly.tools import tool

@tool(
    name="env_tool",
    description="Run an action in a sandboxed Python environment",
    stateful=True,
)
async def env_tool(action: str, context: Context) -> str:
    """
    Run an action in the Python sandbox environment.

    Args:
        action (str): Code or command to run in the sandbox.
        context (Context): Injected rollout context; used to acquire the sandbox resource.
    """
    env = await context.acquire_resource(spec=PythonSandboxSpec, scope="global", backend="local")
    return await env.step(action)
```

## Tool with Schema

```python
@tool(
    name="structured_tool",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"}
        }
    }
)
def structured_tool(query: str, limit: int = 10):
    # Tool implementation
    pass
```
