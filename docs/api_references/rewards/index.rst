.. _rewards_index:

###################
Reward Functions
###################


Overview
========

AgentFly provides a comprehensive suite of reward functions for evaluating agent performance across different tasks and environments. These reward functions are designed to provide meaningful feedback signals for training reinforcement learning agents in various domains.

All reward functions follow a consistent interface pattern using the ``@reward`` decorator and return dictionaries containing reward scores and additional metrics.

Available Reward Functions
==========================

.. toctree::
   :maxdepth: 2

   alfworld_reward
   code_reward  
   math_reward
   qa_reward

Quick Start
===========

.. code-block:: python

   from agentfly.rewards import get_reward_from_name
   
   # Get a specific reward function
   math_reward = get_reward_from_name("math_reward")
   
   # Use the reward function
   result = math_reward("\\boxed{42}", "\\boxed{42}")
   print(result)  # {"reward": 1.0}

Reward Function Categories
=========================

**Environment-Specific Rewards**
    - ALFWorld Episode Reward
    - Code Execution Reward

**Task-Specific Rewards**
    - Math Problem Solving Rewards
    - Question Answering (QA) Rewards

**Format-Aware Rewards**
    - Tool Usage Rewards
    - Thinking Process Rewards
    - Multi-step Reasoning Rewards

Common Patterns
===============

**Basic Reward Usage:**

.. code-block:: python

   @reward(name="my_reward")
   def my_reward(prediction: str, golden_answer: str) -> dict:
       score = calculate_score(prediction, golden_answer)
       return {"reward": score}

**Environment-Based Rewards:**

.. code-block:: python

   @reward(name="env_reward", env_cls=MyEnv, pool_size=8)
   async def env_reward(prediction: str, env: MyEnv) -> dict:
       result = await env.step("get_reward")
       return {"reward": result["reward"]}

**Trajectory-Aware Rewards:**

.. code-block:: python

   @reward(name="trajectory_reward")
   def trajectory_reward(prediction: str, answer: str, trajectory: List[Dict]) -> dict:
       # Analyze agent's trajectory for process rewards
       tool_usage = analyze_tool_usage(trajectory)
       accuracy = check_accuracy(prediction, answer)
       return {"reward": combine_scores(tool_usage, accuracy)}

Technical Implementation
========================

All reward functions use the ``@reward`` decorator which:

- Registers the function in the global reward registry
- Handles environment pooling for stateful rewards
- Provides consistent error handling and logging
- Supports both synchronous and asynchronous execution
- Enables automatic resource management

**Decorator Parameters:**
    - ``name``: Unique identifier for the reward function
    - ``env_cls``: Environment class for stateful rewards (optional)
    - ``pool_size``: Number of environment instances to pool (optional) 