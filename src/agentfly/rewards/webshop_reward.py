from ..core import Context
from ..envs.webshop_text_env import WebShopSpec
from .reward_base import reward


@reward(name="webshop_reward")
async def webshop_reward(
    final_response: str, context: Context, task_id: int
) -> dict:
    """
    Calculates the reward for the WebShop environment based on the environment state.
    Uses the same rollout resource as the webshop tools (context.acquire_resource).

    Args:
        final_response (str): The agent's final response. Not used in this reward function.
        context (Context): Injected rollout context; used to acquire the WebShop resource.
        task_id (int): The identifier for the current task. Used to match with golden answer.

    Returns:
        dict: A dictionary containing the reward (float) and output (str) from the environment step.
    """
    try:
        env = await context.acquire_resource(spec=WebShopSpec, scope="global", backend="local")
        result = await env.step("get_reward", task_id)
        return {
            "reward": result["reward"],
            "output": result["observation"],
        }
    except Exception as e:
        return {
            "reward": 0.0,
            "output": f"Error webshop reward function: {e}",
        }
