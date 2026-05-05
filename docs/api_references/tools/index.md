# Tools

AgentFly provides a comprehensive tool system that enables agents to interact with external systems and APIs. Tools can be stateful (with environment management) or stateless, and support both synchronous and asynchronous execution.

## Structure

- [Tool](tool.md) - Base tool class and decorator
- [Predefined Tools](predefined_tools.md) - Built-in tools

## Basic Tool Definition

```python
from agentfly.tools import tool

--8<-- "src/agentfly/tools/tool_base.py:addition_tool_example"
```

## Stateful Tool with ResourceEngine

```python
from agentfly.core import Context
from agentfly.envs.python_env import PythonSandboxSpec
from agentfly.tools import tool

--8<-- "src/agentfly/tools/src/code/tools.py:code_interpreter_example"
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
