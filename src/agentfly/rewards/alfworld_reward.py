from typing import Any, Dict

from ..core import Context
from ..envs.alfworld_env import ALFWorldSpec
from .reward_base import reward


@reward(name="alfworld_episode_reward")
async def alfworld_episode_reward(prediction: str, context: Context) -> Dict[str, Any]:
    """
    Simple ALFWorld episode reward that checks if the episode is done.
    Uses the same rollout resource as the alfworld tools (context.acquire_resource).
    """
    env = await context.acquire_resource(spec=ALFWorldSpec, scope="rollout", backend="local")
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
