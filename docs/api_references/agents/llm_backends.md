# LLM Backends

## Overview

AgentFly supports multiple LLM backends for text generation, each with their own configuration options.
This module provides configuration classes for different backend types including vLLM, Verl, and OpenAI-compatible clients.
Among them, Verl backend is designed for internal training usage. The Verl backend is the core design that **decouples agent system and rl training**.

## Configuration Classes

### Async VLLM Backend

Configuration for asynchronous vLLM backend with engine arguments:

::: agentfly.agents.llm_backends.backend_configs.AsyncVLLMConfig
    options:
      show_inheritance: true

### Async Verl Backend

Configuration for asynchronous Verl backend:

::: agentfly.agents.llm_backends.backend_configs.AsyncVerlConfig
    options:
      show_inheritance: true

### Client Backend

Configuration for OpenAI-compatible client backends:

::: agentfly.agents.llm_backends.backend_configs.ClientConfig
    options:
      show_inheritance: true

## Backend Implementations

### Base Backend

Abstract base class for all LLM backends:

::: agentfly.agents.llm_backends.llm_backends.LLMBackend
    options:
      show_inheritance: true

### Async VLLM Backend

Asynchronous vLLM implementation for high-performance model inference:

::: agentfly.agents.llm_backends.llm_backends.AsyncVLLMBackend
    options:
      show_inheritance: true

### Async Verl Backend

Asynchronous Verl implementation for distributed model inference:

::: agentfly.agents.llm_backends.llm_backends.AsyncVerlBackend
    options:
      show_inheritance: true

### Client Backend

OpenAI-compatible client backend for remote API inference:

::: agentfly.agents.llm_backends.llm_backends.ClientBackend
    options:
      show_inheritance: true

## Usage Examples

Backends are designed to work together with agents. Here are examples showing how to configure different backends when creating agents:

### Async VLLM Backend

```python
from agentfly.agents import HFAgent
from agentfly.tools import calculator
from agentfly.rewards import math_reward_string_equal
from agentfly.agents.llm_backends import AsyncVLLMConfig

agent = HFAgent(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    tools=[calculator],
    reward_fn=math_reward_string_equal,
    template="qwen2.5",
    backend="async_vllm",
    backend_config=AsyncVLLMConfig(
        pipeline_parallel_size=2,
        data_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8
    )
)
```

### Client Backend (OpenAI-compatible)

```python
from agentfly.agents import HFAgent
from agentfly.tools import calculator
from agentfly.rewards import math_reward_string_equal
from agentfly.agents.llm_backends import ClientConfig

agent = HFAgent(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    tools=[calculator],
    reward_fn=math_reward_string_equal,
    template="qwen2.5",
    backend="client",
    backend_config=ClientConfig(
        base_url="http://localhost:8000/v1",
        api_key="your-api-key",
        max_requests_per_minute=200,
        timeout=300,
        temperature=0.7,
        max_new_tokens=1024
    )
)
```
