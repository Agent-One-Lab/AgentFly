from ..core import Context
from ..envs.webshop_text_env import WebShopSpec
from .reward_base import reward


# --8<-- [start:webshop_reward_example]
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
        score = float(result["reward"])
        return {
            # The dense WebShop score in [0, 1] is the training signal (verl-agent uses it
            # unchanged). ``trajectory/accuracy`` is verl-agent's *success* (score == 1.0),
            # the number their WebShop results report; aggregated per trajectory.
            "reward": score,
            "trajectory/accuracy": float(score == 1.0),
            "output": result["observation"],
        }
    except Exception as e:
        return {
            "reward": 0.0,
            "output": f"Error webshop reward function: {e}",
        }
# --8<-- [end:webshop_reward_example]


@reward(name="webshop_episode_reward")
async def webshop_episode_reward(context: Context, task_id: int) -> dict:
    """WebShop episode reward on verl-agent's scale: ``10`` if the purchase fully satisfies the
    goal (dense score == 1.0), else ``0``.

    This is the reward behind verl-agent's reported WebShop success rates
    (``envs.WebshopWorker.step`` binarizes the env score the same way), so it pairs with the
    ``webshop_browser_action`` tool's per-step reward for GiGPO. ``webshop_reward`` keeps the
    dense score as the training signal instead.

    Returns ``reward`` (0 / 10), ``trajectory/accuracy`` (0 / 1 success, aggregated per
    trajectory) and ``task_score`` (the dense score in [0, 1], a diagnostic).
    """
    from ..tools.src.webshop.tools import WEBSHOP_WON_REWARD

    try:
        env = await context.acquire_resource(spec=WebShopSpec, scope="global", backend="local")
        result = await env.step("get_reward", task_id)
        score = float(result["reward"])
    except Exception:  # noqa: BLE001 — no purchase / env error: an unsuccessful episode
        score = 0.0
    won = float(score == 1.0)
    return {
        "reward": WEBSHOP_WON_REWARD * won,
        "trajectory/accuracy": won,
        "task_score": score,
    }
