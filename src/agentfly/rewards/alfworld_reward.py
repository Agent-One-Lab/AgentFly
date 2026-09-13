from typing import Any, Dict

from ..core import Context
from ..envs.alfworld_env import ALFWorldSpec
from .reward_base import reward


@reward(name="alfworld_episode_reward")
async def alfworld_episode_reward(context: Context) -> Dict[str, Any]:
    """
    ALFWorld episode reward: ``10 * won`` (verl-agent scale), else 0.

    Uses the same rollout resource as the alfworld tools (context.acquire_resource).
    We read the stored terminal `won` flag instead of taking another env step —
    ALFWorld only pays out reward on the winning transition, so an extra step here
    would return 0 and discard the actual outcome. The ``10 *`` scale matches
    verl-agent's ``compute_reward`` (``10.0 * float(info['won'])``); with the fixed 0.1
    invalid-action penalty this keeps the penalty a ~1% nudge rather than a 10% force.
    """
    env = await context.acquire_resource(spec=ALFWorldSpec, scope="global", backend="local")
    info = await env.get_info()
    won = bool(info.get("won", False)) if info else False

    # ``reward`` is the training signal (10 * won, verl-agent scale). ``trajectory/accuracy``
    # is the 0/1 task-success flag under the ``trajectory/`` namespace, so it's aggregated
    # per episode (reduced by traj_uid) -> the per-trajectory success rate (0-1), the metric
    # verl-agent reports. (The raw ``reward`` mean would be 10x that and length-weighted.)
    return {
        "reward": 10.0 * float(won),
        "trajectory/accuracy": float(won),
    }
