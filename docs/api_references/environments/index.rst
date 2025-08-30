.. _environments_index:

###################
Environments API Reference
###################


Overview
========

AgentFly provides a comprehensive environment system for managing agent interactions with external systems. Environments can be stateful, support Docker containers, and provide async interfaces for high-performance execution.

Base Environment
================

.. toctree::
   :maxdepth: 2

   environment/

Core Classes
===========

BaseEnv
-------

The foundation class for all environments:

.. autoclass:: agentfly.envs.env_base.BaseEnv
   :members:
   :show-inheritance:
   :special-members: __init__

SupportsDocker
--------------

Mixin for Docker-based environments:

.. autoclass:: agentfly.envs.env_base.SupportsDocker
   :members:
   :show-inheritance:

Pre-built Environments
=====================

ALFWorld Environment
--------------------

ALFWorld text-based environment:

.. autoclass:: agentfly.envs.alfworld_env.ALFWorldEnv
   :members:
   :show-inheritance:

ScienceWorld Environment
------------------------

ScienceWorld experiment environment:

.. autoclass:: agentfly.envs.scienceworld_env.ScienceWorldEnv
   :members:
   :show-inheritance:

Python Sandbox Environment
--------------------------

Python code execution environment:

.. autoclass:: agentfly.envs.python_env.PythonSandboxEnv
   :members:
   :show-inheritance:

WebShop Text Environment
------------------------

WebShop text-based interface:

.. autoclass:: agentfly.envs.webshop_text_env.WebAgentTextEnv
   :members:
   :show-inheritance:

Environment Management
=====================

Environment Manager
------------------

Centralized environment management:

.. autoclass:: agentfly.envs.manager.env_manager.EnvironmentManager
   :members:
   :show-inheritance:

Resource Management
------------------

Global environment resource tracking:

.. autofunction:: agentfly.envs.manager.resource.GLOBAL_ENVS

Usage Examples
=============

Basic Environment
----------------

.. code-block:: python

   from agentfly.envs import BaseEnv
   
   class SimpleEnv(BaseEnv):
       async def start(self):
           # Initialize resources
           pass
       
       async def reset(self):
           # Reset to initial state
           return "initial_state"
       
       async def step(self, action):
           # Execute action
           return f"result_of_{action}"
       
       async def aclose(self):
           # Clean up resources
           pass
       
       @staticmethod
       async def acquire():
           return SimpleEnv()

Docker-based Environment
-----------------------

.. code-block:: python

   from agentfly.envs import BaseEnv, SupportsDocker
   
   class DockerEnv(BaseEnv, SupportsDocker):
       def __init__(self):
           super().__init__()
           self.image = "my-image:latest"
           self.container = None
       
       async def start(self):
           self.container = await self.start_container(
               image=self.image,
               runtime="runc",
               cpu=2,
               mem="2g"
           )
       
       async def reset(self):
           # Reset container state
           return "reset_state"
       
       async def step(self, action):
           # Execute action in container
           return f"container_result_{action}"
       
       async def aclose(self):
           if self.container:
               await self.stop_container(self.container)

Environment with Tools
---------------------

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
