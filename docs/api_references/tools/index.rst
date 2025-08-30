.. _tools_index:

###################
Tools API Reference
###################


Overview
========

AgentFly provides a comprehensive tool system that enables agents to interact with external systems and APIs. Tools can be stateful (with environment management) or stateless, and support both synchronous and asynchronous execution.

Base Tool
=========

.. toctree::
   :maxdepth: 2

   tool

Core Classes
===========

Tool
----

The main tool wrapper class:

.. autoclass:: agentfly.tools.tool_base.Tool
   :members:
   :show-inheritance:
   :special-members: __call__

Tool Decorator
-------------

The main decorator for creating tools:

.. autofunction:: agentfly.tools.tool_base.tool

Utility Functions
================

Tool Registration
----------------

.. autofunction:: agentfly.tools.register_tool

Tool Retrieval
--------------

.. autofunction:: agentfly.tools.get_tools_from_names

Pre-built Tools
==============

Code Tools
----------

.. autofunction:: agentfly.tools.src.code.tools.code_interpreter

Search Tools
------------

.. autofunction:: agentfly.tools.src.search.google_search.google_search_serper

.. autofunction:: agentfly.tools.src.search.dense_retriever.dense_retrieve

.. autofunction:: agentfly.tools.src.search.async_dense_retriever.asyncdense_retrieve

ALFWorld Tools
--------------

.. autofunction:: agentfly.tools.src.alfworld.tools.alfworld_step

.. autofunction:: agentfly.tools.src.alfworld.tools.alfworld_reset

.. autofunction:: agentfly.tools.src.alfworld.tools.alfworld_get_task_objective

.. autofunction:: agentfly.tools.src.alfworld.tools.alfworld_get_admissible_commands

WebShop Tools
-------------

.. autofunction:: agentfly.tools.src.webshop.tools.webshop_browser

ScienceWorld Tools
------------------

.. autofunction:: agentfly.tools.src.scienceworld.tools.scienceworld_explorer

ReAct Tools
-----------

.. autofunction:: agentfly.tools.src.react.tools.answer_qa

.. autofunction:: agentfly.tools.src.react.tools.answer_math

Utility Tools
-------------

.. autofunction:: agentfly.tools.src.calculate.tools.calculator

.. autofunction:: agentfly.tools.src.ui.tools.pyautogui_code_generator

Usage Examples
=============

Basic Tool Definition
--------------------

.. code-block:: python

   from agentfly.tools import tool
   
   @tool(name="calculator", description="Calculate mathematical expressions")
   def calculator(expression: str):
       try:
           result = eval(expression)
           return str(result)
       except Exception as e:
           return f"Error: {str(e)}"

Stateful Tool with Environment
-----------------------------

.. code-block:: python

   from agentfly.tools import tool
   from agentfly.envs import BaseEnv
   
   class MyEnv(BaseEnv):
       # Environment implementation
       pass
   
   @tool(name="env_tool", env_cls=MyEnv, pool_size=4)
   async def env_tool(action: str, env: MyEnv):
       result = await env.step(action)
       return result

Tool with Schema
---------------

.. code-block:: python

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
