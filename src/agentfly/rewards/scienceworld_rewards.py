"""ScienceWorld reward: ``scienceworld_reward`` gives 10 when the rollout's ScienceWorld task is completed (score 1), else 0."""

from ..core import Context
from ..envs.scienceworld_env import ScienceWorldSpec
from .reward_base import reward


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
    # With context folding, the trajectory is a list of segments (each a list of
    # messages), so len(trajectory) would count segments rather than messages.
    if context.trajectory_format == "segmented":
        num_messages = sum(len(segment) for segment in trajectory)
    else:
        num_messages = len(trajectory)
    if num_messages < 4:
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

    reward = acc * 10.0

    return {
        "reward": reward,
        "acc": acc,
        "score": result["reward"],
        "output": result["observation"],
    }
