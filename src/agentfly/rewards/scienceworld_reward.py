from ..core import Context
from ..envs.scienceworld_env import ScienceWorldSpec
from .reward_base import reward
from typing import List


@reward(name="scienceworld_reward")
async def scienceworld_reward(final_response: str, context: Context) -> dict:
    """
    Computes the reward for a given prediction in the ScienceWorld environment.
    Uses the same rollout resource as the scienceworld tools (context.acquire_resource).

    Args:
        final_response (str): The agent's final response. Not used in this reward function.
        context (Context): Injected rollout context; used to acquire the ScienceWorld resource.

    Returns:
        dict: A dictionary containing the reward and the observation output after taking the 'get_reward' step.
    """
    trajectory = context.trajectory
    if len(trajectory) < 4:
        return {
            "reward": 0.0,
            "acc": 0.0,
            "output": "Not enough steps to get reward",
        }

    env = await context.acquire_resource(
        spec=ScienceWorldSpec,
        scope="global",
        backend="local",
    )

    result = await env.step("get_reward")
    if result["reward"] >= 1:
        acc = 1.0
    else:
        acc = 0.0

    return {
        "reward": result["reward"],
        "acc": acc,
        "output": result["observation"],
    }
