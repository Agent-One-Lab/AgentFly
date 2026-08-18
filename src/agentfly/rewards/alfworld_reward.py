from typing import Any, Dict

from ..core import Context
from ..envs.alfworld_env import ALFWorldSpec
from .reward_base import reward


@reward(name="alfworld_episode_reward")
async def alfworld_episode_reward(context: Context) -> Dict[str, Any]:
    """
    ALFWorld episode reward: 1 if the task was solved, else 0.

    Uses the same rollout resource as the alfworld tools (context.acquire_resource).
    We read the stored terminal `won` flag instead of taking another env step —
    ALFWorld only pays out reward on the winning transition, so an extra step here
    would return 0 and discard the actual outcome.
    """
    env = await context.acquire_resource(spec=ALFWorldSpec, scope="global", backend="local")
    info = await env.get_info()
    won = bool(info.get("won", False)) if info else False

    return {
        "reward": float(won),
    }
