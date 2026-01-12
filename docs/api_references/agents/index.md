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
from agentfly.tools import get_tools_from_names

# Create a ReactAgent with tools
agent = ReactAgent(
    model_name_or_path="gpt2",
    tools=get_tools_from_names(["calculator", "google_search"]),
    template="react"
)
```

### Using AutoAgent

```python
from agentfly.agents import AutoAgent

# Create agent from config
config = {
    "agent_type": "react",
    "model_name_or_path": "gpt2",
    "template": "react",
    "tools": ["calculator"]
}
agent = AutoAgent.from_config(config)
```

### Custom Agent

```python
from agentfly.agents import BaseAgent

class CustomAgent(BaseAgent):
    def parse(self, response):
        # Custom parsing logic
        pass
    
    def generate(self, messages):
        # Custom generation logic
        pass
```

