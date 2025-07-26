.. _alfworld_environment:

ALFWorld Environment
====================

The ALFWorldEnv class provides a Python interface to interact with ALFWorld environments running in Docker containers.

Class Reference
---------------

.. currentmodule:: agents.envs.alfworld_env
.. autoclass:: agents.envs.alfworld_env.ALFWorldEnv
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from agents.agents.envs.alfworld_env import ALFWorldEnv
    
    # Create environment with default settings
    env = ALFWorldEnv()
    await env.start()
    
    # Reset to start a new episode
    obs, info = await env.reset()
    print(f"Initial observation: {obs}")
    
    # Take an action
    obs, reward, done, info = await env.step("go to kitchen")
    print(f"Reward: {reward}, Done: {done}")
    
    # Clean up
    await env.aclose()

Custom Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create environment with custom settings
    env = ALFWorldEnv(
        image="custom/alfworld-env:latest",
        cpu=4,
        mem="4g",
        train_eval="valid_seen",
        max_episodes=100
    )
    await env.start()
    
    # Reset to a specific task
    obs, info = await env.reset(
        env_args={"task_id": "trial_T20190907_212755_456877"}
    )



Configuration Parameters
-----------------------

The ALFWorldEnv class accepts the following configuration parameters:

Docker Settings
~~~~~~~~~~~~~~

* **image**: Docker image to use (default: ``bitalov/alfworld-http-env-3:latest``)
* **runtime**: Docker runtime (default: ``runc``)
* **cpu**: Number of CPU cores (default: ``2``)
* **mem**: Memory allocation (default: ``2g``)

Environment Settings
~~~~~~~~~~~~~~~~~~~

* **train_eval**: Data split to use (default: ``train``)
* **batch_size**: Batch size for ALFWorld (default: ``1``)
* **config_path**: Optional custom ALFWorld config file path
* **max_episodes**: Maximum episodes before restart (default: ``50``)

Network Settings
~~~~~~~~~~~~~~~

* **host_ip**: Host IP for port mapping (default: ``127.0.0.1``)
* **container_port**: Container internal port (default: ``8000``)
* **start_timeout**: Startup timeout in seconds (default: ``120.0``) 