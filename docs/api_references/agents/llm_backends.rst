LLM Backends
============================

Overview
-----------------------------

AgentFly supports multiple LLM backends for text generation, each with their own configuration options.
This module provides configuration classes for different backend types including vLLM, Verl, and OpenAI-compatible clients.
Among them, Verl backend is designed for internal training usage. The Verl backend is the core design that **decouples agent system and rl training**.

Configuration Classes
-----------------------------


Async VLLM Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration for asynchronous vLLM backend with engine arguments:

.. autoclass:: agentfly.agents.llm_backends.backend_configs.AsyncVLLMConfig
   :show-inheritance:
   :special-members: !__init__


Async Verl Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration for asynchronous Verl backend:

.. autoclass:: agentfly.agents.llm_backends.backend_configs.AsyncVerlConfig
   :show-inheritance:
   :special-members: !__init__

Client Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration for OpenAI-compatible client backends:

.. autoclass:: agentfly.agents.llm_backends.backend_configs.ClientConfig
   :show-inheritance:
   :special-members: !__init__

Backend Implementations
------------------------------

Base Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract base class for all LLM backends:

.. autoclass:: agentfly.agents.llm_backends.llm_backends.LLMBackend
   :members:
   :show-inheritance:


Async VLLM Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asynchronous vLLM implementation for high-performance model inference:

.. autoclass:: agentfly.agents.llm_backends.llm_backends.AsyncVLLMBackend
   :members:
   :show-inheritance:

Async Verl Backend
~~~~~~~~~~~~~~~~~~

Asynchronous Verl implementation for distributed model inference:

.. autoclass:: agentfly.agents.llm_backends.llm_backends.AsyncVerlBackend
   :members:
   :show-inheritance:

Client Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenAI-compatible client backend for remote API inference:

.. autoclass:: agentfly.agents.llm_backends.llm_backends.ClientBackend
   :members:
   :show-inheritance:


Usage Examples
------------------------------

Backends are designed to work together with agents. Here are examples showing how to configure different backends when creating agents:


Async VLLM Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

Client Backend (OpenAI-compatible)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

