from ..envs.alfworld_env import ALFWorldEnv
from .reward_base import reward
from typing import Dict, Any


@reward(
    name="alfworld_episode_reward",
    env_cls=ALFWorldEnv,
    pool_size=8,  # Reasonable pool size for ALFWorld environments
)
async def alfworld_episode_reward(prediction: str, env: ALFWorldEnv) -> Dict[str, Any]:
    """
    Simple ALFWorld episode reward that checks if the episode is done.
    """

    # Step with empty action to get current state
    print("------Reward--------------")
    obs, reward_val, done, info = await env.step("")
    if reward_val is None:
        print("Reward is None")
        reward_val = 0.0
    print(reward_val)
    print("--------------\n")

    return {
        "reward": reward_val,
    }
