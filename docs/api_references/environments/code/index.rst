.. _code_index:

##################
Code Environment
##################

.. contents::
   :local:
   :depth: 2

Overview
========

The Code Environment provides a secure, isolated Python sandbox for executing untrusted code snippets. It uses Docker containers with strict resource limits and security controls to safely run Python code and return execution results.

The environment is built on the ``PythonSandboxEnv`` class, which spawns Docker containers running a FastAPI-based HTTP server for code execution. Each environment instance provides an isolated Python interpreter with configurable memory, CPU, and timeout limits.

.. _TextWorld: https://www.microsoft.com/en-us/research/project/textworld/

Components
==========

The Code environment includes the following components:

.. toctree::
   :maxdepth: 2

   environment
   http_server
   tools
   rewards

Quick Start
===========

For most use cases, you can use the PythonSandboxEnv class directly:

.. code-block:: python

   from agentfly.envs.python_env import PythonSandboxEnv
   
   # Create and start the environment
   env = PythonSandboxEnv()
   await env.start()
   
   # Execute Python code
   result = await env.step("print('Hello, World!')")
   print(result)  # Output: Hello, World!
   
   # Maintain state between executions
   await env.step("x = 42")
   result = await env.step("print(x)")
   print(result)  # Output: 42

For HTTP-based integration, see the :doc:`http_server` documentation.

Key Features
============

* **Docker-based Isolation**: Runs in secure containers with resource limits
* **State Persistence**: Variables persist between code executions within the same session
* **HTTP API**: RESTful interface for language-agnostic integration
* **Security Controls**: Read-only filesystem, dropped capabilities, process limits
* **Timeout Protection**: Configurable execution timeouts to prevent infinite loops
* **Resource Management**: CPU and memory limits for safe execution
* **Automatic Recovery**: Built-in error handling and container restart capabilities 