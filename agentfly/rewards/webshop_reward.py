from ..envs.webshop_text_env import WebAgentTextEnv
from .reward_base import reward

@reward(name="webshop_reward", env_cls=WebAgentTextEnv, pool_size=8)
async def webshop_reward(final_response: str, env: WebAgentTextEnv, task_id: int) -> dict:
    """
    Calculates the reward for the WebShop environment based on the environment state. Match the purchased product with the golden answer characteristics.
    Actual logic for reward calculation is in the environment and partially in step method of the environment.
    Adapted from https://arxiv.org/pdf/2207.01206

    Args:
        final_response (str): The agent's final response. Not used in this reward function.
        env (WebAgentTextEnv): The environment instance for the WebShop task.
        task_id (int): The identifier for the current task. Used to match with golden answer.

    Returns:
        dict: A dictionary containing the reward (float) and output (str) from the environment step. If an error occurs, returns a reward of 0.0 and an error message as output.
    """
    try:
        result = await env.step('get_reward', task_id)
        return {
            "reward": result["reward"],
            "output": result["observation"],
        }
    except Exception as e:
        return {
            "reward": 0.0,
            "output": f"Error webshop reward function: {e}",
        }