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

## Stateful Tool with Environment

```python
from agentfly.tools import tool
from agentfly.envs import BaseEnv

class MyEnv(BaseEnv):
    # Environment implementation
    pass

@tool(name="env_tool", env_cls=MyEnv, pool_size=4)
async def env_tool(action: str, env: MyEnv):
    result = await env.step(action)
    return result
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
