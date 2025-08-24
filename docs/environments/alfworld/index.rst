.. _alfworld_index:

######################
ALFWorld Environment
######################

.. contents::
   :local:
   :depth: 2

Overview
========

ALFWorld (Action Learning From World) is an interactive text-based environment that simulates a household setting. It is built upon the `TextWorld`_ framework and is designed to test an agent's ability to perform complex, multi-step tasks by following natural language instructions.

.. _TextWorld: https://www.microsoft.com/en-us/research/project/textworld/

Components
==========

The ALFWorld environment includes the following components:

.. toctree::
   :maxdepth: 2

   environment
   http_server
   tools
   rewards

Quick Start
===========

For most use cases, you can use the ALFWorldEnv class directly:

.. code-block:: python

   from agentfly.envs.alfworld_env import ALFWorldEnv
   
   # Create and start the environment
   env = ALFWorldEnv()
   await env.start()
   
   # Reset to start a new episode
   obs, info = await env.reset()
   
   # Take actions
   obs, reward, done, info = await env.step("go to kitchen")

For HTTP-based integration, see the :doc:`http_server` documentation.

Key Features
============

* **Docker-based Isolation**: Runs in containers for consistent environments
* **Task Selection**: Support for both random and specific task selection
* **HTTP API**: RESTful interface for language-agnostic integration
* **Comprehensive State Access**: Full access to observations, admissible commands, and environment state
* **Automatic Recovery**: Built-in error handling and container restart capabilities 