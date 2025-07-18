AgentFly
=====================
AgentFly is a scalable and extensible Agent-RL framework designed to empower LM agents with a variety of RL algorithms. The framework supports multi-turn interactions by adapting traditional RL methods with token-level masking. It features a decorator-based interface for defining tools and reward functions, enabling seamless extension and ease of use. To support high-throughput training, we implement asynchronous execution of tool calls and reward computations, and design a centralized resource management system for scalable environment coordination. We also provide a suite of prebuilt tools and environments, demonstrating the framework's effectiveness through successful agent training across multiple tasks.

---------------------
.. _Contents:

.. toctree::
    :maxdepth: 2
    :caption: Quick Start

    start/installation
    start/training_example

.. toctree::
    :maxdepth: 2
    :caption: Features

    features/agent_rollout


.. toctree::
   :maxdepth: 2
   :caption: Environments

   environments/alfworld/index
   environments/code/index
   environments/retrieval/index

.. toctree::
    :maxdepth: 2
    :caption: Rewards

    rewards/index

.. toctree::
    :maxdepth: 2
    :caption: Classes

    classes/agent
    classes/tool
    classes/reward
    classes/environment

---------------------