from ..envs.scienceworld_env import ScienceWorldEnv
from .reward_base import reward

@reward(name="scienceworld_reward", env_cls=ScienceWorldEnv, pool_size=8)
async def scienceworld_reward(prediction: str, env: ScienceWorldEnv) -> dict:
    """
    Computes the reward for a given prediction in the ScienceWorld environment.
    Actual logic for reward calculation is in the environment and partially in step method of the environment.
    The reward is the highest score achieved in the task (which subgoal is reached, 1 if the full task is completed).
    Adapted and modified from https://github.com/allenai/ScienceWorld/tree/main

    Args:
        prediction (str): The predicted action or answer (currently unused in this function).
        env (ScienceWorldEnv): The ScienceWorld environment instance.

    Returns:
        dict: A dictionary containing the reward and the observation output after taking the 'get_reward' step.
    """
    result = await env.step('get_reward')
    return {
        "reward": result["reward"],
        "output": result["observation"],
    }
