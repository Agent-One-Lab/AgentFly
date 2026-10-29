# LLM Backends

## Overview

AgentFly supports multiple LLM backends for text generation, each with their own configuration options.
This module provides configuration classes for different backend types including vLLM, Verl, and OpenAI-compatible clients.
Among them, Verl backend is designed for internal training usage. The Verl backend is the core design that **decouples agent system and rl training**.

## Configuration Classes

### Async VLLM Backend

Configuration for asynchronous vLLM backend with engine arguments:

::: agentfly.utils.llm_backends.backend_configs.AsyncVLLMConfig
    options:
      show_inheritance: true

### Async Verl Backend

Configuration for asynchronous Verl backend:

::: agentfly.utils.llm_backends.backend_configs.AsyncVerlConfig
    options:
      show_inheritance: true

### Client Backend

Configuration for OpenAI-compatible client backends:

::: agentfly.utils.llm_backends.backend_configs.ClientConfig
    options:
      show_inheritance: true

## Backend Implementations

### Base Backend

Abstract base class for all LLM backends:

::: agentfly.utils.llm_backends.llm_backends.LLMBackend
    options:
      show_inheritance: true

### Async VLLM Backend

Asynchronous vLLM implementation for high-performance model inference:

::: agentfly.utils.llm_backends.llm_backends.AsyncVLLMBackend
    options:
      show_inheritance: true

### Async Verl Backend

Asynchronous Verl implementation for distributed model inference:

::: agentfly.utils.llm_backends.llm_backends.AsyncVerlBackend
    options:
      show_inheritance: true

### Client Backend

OpenAI-compatible client backend for remote API inference:

::: agentfly.utils.llm_backends.llm_backends.ClientBackend
    options:
      show_inheritance: true

## Usage Examples

Agents take the backend as `backend_config`: a dict with a `"backend"` key (`"async_vllm"`,
`"client"`, or `"async_verl"`, which training sets up) plus that backend's settings. Generation
settings such as temperature are passed per call, through `agent.run(generation_config=...)`.

### Async vLLM backend

Runs the model in-process with vLLM. Every other key is passed to vLLM's
[`AsyncEngineArgs`](https://docs.vllm.ai/en/latest/configuration/engine_args.html):

```python
from agentfly.agents import HFAgent
from agentfly.tools import calculator
from agentfly.rewards import math_equal_reward_tool

agent = HFAgent(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    tools=[calculator],
    reward_fn=math_equal_reward_tool,
    template="qwen2.5",
    backend_config={
        "backend": "async_vllm",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 16384,
    },
)
```

Scripts that use this backend need an `if __name__ == "__main__":` guard (see
[Build an Agent](../../start/first_agent.md#a-complete-script)).

### Client backend (OpenAI-compatible)

Sends requests to a server, for example one started with `agentfly deploy` or `vllm serve`, or a
hosted API. The server applies its own chat template, so don't pass `template`:

```python
from agentfly.agents import HFAgent
from agentfly.tools import calculator
from agentfly.rewards import math_equal_reward_tool

agent = HFAgent(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",  # the model name the server expects
    tools=[calculator],
    reward_fn=math_equal_reward_tool,
    backend_config={
        "backend": "client",
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "max_requests_per_minute": 200,
        "timeout": 300,
    },
)
result = await agent.run(
    messages={"messages": [{"role": "user", "content": "What is 3 * 7?"}], "answer": "21"},
    max_turns=3,
    group_size=1,
    generation_config={"temperature": 0.7, "max_tokens": 1024},
)
```
