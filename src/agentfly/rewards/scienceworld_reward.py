from ..envs.scienceworld_env import ScienceWorldEnv
from .reward_base import reward


@reward(name="scienceworld_reward", env_cls=ScienceWorldEnv, pool_size=24)
async def scienceworld_reward(final_response: str, env: ScienceWorldEnv) -> dict:
    """
    Computes the reward for a given prediction in the ScienceWorld environment.
    Actual logic for reward calculation is in the environment and partially in step method of the environment.
    The reward is the highest score achieved in the task (which subgoal is reached, 1 if the full task is completed).
    Adapted and modified from https://github.com/allenai/ScienceWorld/tree/main

    Args:
        final_response (str): The agent's final response. Not used in this reward function.

    Returns:
        dict: A dictionary containing the reward and the observation output after taking the 'get_reward' step.
    """
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
