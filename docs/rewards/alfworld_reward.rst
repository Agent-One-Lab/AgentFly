.. _alfworld_reward:

ALFWorld Episode Reward
========================

.. currentmodule:: agents.agents.rewards.alfworld_reward

.. autofunction:: alfworld_episode_reward

Function Signature
------------------

.. code-block:: python

    async def alfworld_episode_reward(prediction: str, env: ALFWorldEnv) -> Dict[str, Any]

Description
-----------

Evaluates agent performance in ALFWorld tasks by checking the episode completion status and reward from the environment state.

**Parameters:**
    - **prediction** (str): Agent's predicted action or response (not directly used)
    - **env** (ALFWorldEnv): ALFWorld environment instance

**Returns:**
    Dict[str, Any]: Dictionary containing:
        - **reward** (float): Environment reward value for current state

**Decorator Configuration:**
    - **name**: "alfworld_episode_reward"
    - **env_cls**: ALFWorldEnv  
    - **pool_size**: 8

Technical Details
-----------------

**Implementation:**
    - Steps environment with empty action to get current state
    - Extracts reward value from environment response
    - Handles None reward values by defaulting to 0.0
    - Provides debug output for reward values

**Use Cases:**
    - Evaluating task completion in household environments
    - Training agents on multi-step instruction following
    - Measuring progress in text-based interactive environments

**Example Usage:**

.. code-block:: python

    from agents.agents.rewards import get_reward_from_name
    from agents.agents.envs import ALFWorldEnv
    
    # Get reward function
    reward_fn = get_reward_from_name("alfworld_episode_reward")
    
    # Create environment
    env = ALFWorldEnv()
    await env.start()
    
    # Get reward for current state
    result = await reward_fn("take apple", env=env)
    print(result)  # {"reward": 0.0} or {"reward": 1.0} if task completed

**Environment Integration:**
    - Requires active ALFWorld environment instance
    - Uses environment's internal reward mechanism
    - Suitable for episodic task evaluation 