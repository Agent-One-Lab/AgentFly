# Agents API Reference

## Overview

AgentFly provides a comprehensive agent system with a base class and specialized implementations for different use cases. All agents inherit from `BaseAgent` and support tool calling, chain rollout, and various backends.

## Structure

- [Agent](agent.md) - Base agent class and implementations
- [LLM Backends](llm_backends.md) - Language model backends
- [Rollout](rollout.md) - Agent rollout strategies

## Usage Examples

### Basic Agent Creation

```python
from agentfly.agents import ReactAgent
from agentfly.tools import calculator

# ReactAgent writes the tool list into its own ReAct prompt, so it uses a template
# that does not add tools to the system message.
agent = ReactAgent(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    tools=[calculator],
    template="qwen2.5-no-system-tool",
    backend_config={"backend": "async_vllm"},
)
```

### Using AutoAgent

`AutoAgent` builds an agent from a config dict, as training does. Here tools and the reward are
given by registered name:

```python
from agentfly.agents import AutoAgent

config = {
    "agent_type": "react",
    "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
    "template": "qwen2.5-no-system-tool",
    "tools": ["calculator"],
    "reward_name": "math_equal_reward_tool",
    "backend_config": {"backend": "async_vllm"},
}
agent = AutoAgent.from_config(config)
```

### Custom Agent

Subclass `BaseAgent` and override `parse`, which turns raw model outputs into assistant messages.
Put any tool calls the response asks for in `tool_calls`; the rollout runs them:

```python
from agentfly.agents import BaseAgent


class CustomAgent(BaseAgent):
    def parse(self, responses, context=None, **kwargs):
        messages = []
        for response in responses:
            tool_calls = []  # e.g. [{"id": None, "type": "function",
            #                         "function": {"name": "calculator", "arguments": {"expression": "1+1"}}}]
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": response}], "tool_calls": tool_calls}
            )
        return messages
```

See `ReactAgent` (`agentfly/agents/specialized/react_agent.py`) for a complete example.
