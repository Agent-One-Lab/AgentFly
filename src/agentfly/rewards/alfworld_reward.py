from typing import Any, Dict

from ..core import Context
from ..envs.alfworld_env import ALFWorldSpec
from .reward_base import reward


@reward(name="alfworld_episode_reward")
async def alfworld_episode_reward(context: Context) -> Dict[str, Any]:
    """
    Simple ALFWorld episode reward that checks if the episode is done.
    Uses the same rollout resource as the alfworld tools (context.acquire_resource).
    """
    env = await context.acquire_resource(spec=ALFWorldSpec, scope="global", backend="local")
    _obs, reward_val, _done, _info = await env.step("")
    if reward_val is None:
        reward_val = 0.0

    return {
        "reward": reward_val,
    }
