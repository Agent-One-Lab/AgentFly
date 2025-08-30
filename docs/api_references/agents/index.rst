.. _agents_index:

###################
Agents API Reference
###################

Overview
========

AgentFly provides a comprehensive agent system with a base class and specialized implementations for different use cases. All agents inherit from :py:class:`BaseAgent` and support tool calling, chain rollout, and various backends.

Base Agent
==========

.. toctree::
   :maxdepth: 2

   agent
   llm_backends

Core Classes
===========

BaseAgent
---------

The foundation class for all agents in AgentFly:

.. autoclass:: agentfly.agents.agent_base.BaseAgent
   :members:
   :show-inheritance:
   :special-members: __init__

AutoAgent
---------

Factory class for automatic agent creation:

.. autoclass:: agentfly.agents.auto.AutoAgent
   :members:
   :show-inheritance:

Specialized Agents
==================

ReactAgent
----------

ReAct-style agent for reasoning and tool use:

.. autoclass:: agentfly.agents.react.react_agent.ReactAgent
   :members:
   :show-inheritance:

CodeAgent
---------

Specialized agent for code generation and execution:

.. autoclass:: agentfly.agents.specialized.code_agent.CodeAgent
   :members:
   :show-inheritance:

ThinkAgent
----------

Agent that uses thinking steps before taking actions:

.. autoclass:: agentfly.agents.specialized.think_agent.ThinkAgent
   :members:
   :show-inheritance:

GUIAgent
---------

Agent for GUI automation tasks:

.. autoclass:: agentfly.agents.specialized.gui_agent.GUIAgent
   :members:
   :show-inheritance:

HFAgent
--------

Hugging Face model-based agent:

.. autoclass:: agentfly.agents.specialized.hf_agent.HFAgent
   :members:
   :show-inheritance:

OpenAIAgent
-----------

OpenAI API-based agent:

.. autoclass:: agentfly.agents.specialized.openai_agent.OpenAIAgent
   :members:
   :show-inheritance:

Chain Generation
===============

ChainRollout
------------

Base class for chain-based generation:

.. autoclass:: agentfly.agents.chain.chain_base.ChainRollout
   :members:
   :show-inheritance:

Usage Examples
=============

Basic Agent Creation
-------------------

.. code-block:: python

   from agentfly.agents import ReactAgent
   from agentfly.tools import get_tools_from_names
   
   # Create a ReactAgent with tools
   agent = ReactAgent(
       model_name_or_path="gpt2",
       tools=get_tools_from_names(["calculator", "google_search"]),
       template="react"
   )

Using AutoAgent
--------------

.. code-block:: python

   from agentfly.agents import AutoAgent
   
   # Create agent from config
   config = {
       "agent_type": "react",
       "model_name_or_path": "gpt2",
       "template": "react",
       "tools": ["calculator"]
   }
   agent = AutoAgent.from_config(config)

Custom Agent
-----------

.. code-block:: python

   from agentfly.agents import BaseAgent
   
   class CustomAgent(BaseAgent):
       def parse(self, response):
           # Custom parsing logic
           pass
       
       def generate(self, messages):
           # Custom generation logic
           pass
