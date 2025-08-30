.. _alfworld_rewards:

ALFWorld Rewards
================

The ALFWorld reward system provides sparse, outcome-based evaluation for task completion. Rewards are binary (1 for task completion, 0 for incomplete tasks) and focus on successful episode completion rather than incremental progress.

Reward Functions Reference
--------------------------

.. currentmodule:: agents.rewards.alfworld_reward

alfworld_episode_reward
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: agents.rewards.alfworld_reward.alfworld_episode_reward

**Function Signature:**

.. code-block:: python

    async def alfworld_episode_reward(prediction: str, env: ALFWorldEnv) -> Dict[str, Any]

**Description:** Evaluate the reward for an agent's action in the ALFWorld environment

**Parameters:**
    - **prediction** (str): The agent's predicted action or response (not directly used in evaluation but required by the reward interface)
    - **env** (ALFWorldEnv): The ALFWorld environment instance to evaluate the reward from

**Returns:**
    Dict[str, Any]: A dictionary containing:
        - reward (float): The numerical reward value for the current state. Positive values indicate progress or task completion, zero indicates no progress, negative values indicate invalid actions or moving away from the goal

Usage with ReactAgent
---------------------

Real Example from Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how the ALFWorld reward is used with a ReactAgent:

.. code-block:: python

    from agentfly.agents.react.react_agent import ReactAgent
    from agentfly.rewards import alfworld_episode_reward

    # Create ReactAgent with ALFWorld reward function
    react_agent = ReactAgent(
        "Qwen/Qwen2.5-7B-Instruct",
        tools=tools,
        reward_fn=alfworld_episode_reward,  # Reward function integrated
        template="qwen-chat",
        task_info=task_info,
        backend="async_vllm",
        debug=True
    )

    # After agent execution, rewards are automatically calculated
    await react_agent.run_async(
        max_steps=12,
        start_messages=messages,
        num_chains=4
    )

    # Get rewards for all trajectories
    rewards, other_values = react_agent.rewards
    print(f"Rewards: {rewards}")

Simple Direct Usage
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Simple reward function usage
    reward_result = await alfworld_episode_reward(
        prediction="take apple", 
        env=env
    )
    print(f"Reward: {reward_result['reward']}")
    
    # Get reward by name
    from agentfly.rewards import get_reward_from_name
    reward_fn = get_reward_from_name("alfworld_episode_reward")
    result = await reward_fn("take apple", env)
    print(result)

Resource Management
------------------

Environment Pool Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reward function is configured with:

* **Pool size**: 8 ALFWorld environments
* **Environment class**: ALFWorldEnv  
* **Concurrent evaluations**: Supported through pool management

Reward Values Interpretation
---------------------------

ALFWorld rewards typically follow these patterns:

* **1.0**: Task successfully completed
* **0.0**: No progress made, neutral action
* **-0.1**: Invalid action or syntax error
* **Sparse rewards**: Most actions return 0.0, with positive rewards only on task completion

Debugging Output
~~~~~~~~~~~~~~~

The reward function includes debug output to help track evaluation:

.. code-block:: text

    ------Reward--------------
    1.0
    --------------

This output shows when the reward is calculated and the actual reward value received from the environment. 