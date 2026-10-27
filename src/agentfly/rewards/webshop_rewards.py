"""WebShop rewards from the environment's purchase score.

``webshop_reward`` is verl-agent's scale: 10 when the purchase fully matches the goal, otherwise 0.
``webshop_score_reward`` returns the dense score in [0, 1] for the ``task_id`` goal instead. Both
log success (score 1) as ``trajectory/accuracy``.
"""

from ..core import Context
from ..envs.webshop_text_env import WebShopSpec
from .reward_base import reward


# --8<-- [start:webshop_reward_example]
@reward(name="webshop_reward")
async def webshop_reward(context: Context, task_id: int) -> dict:
    """WebShop episode reward on verl-agent's scale: ``10`` if the purchase fully satisfies the
    goal (dense score == 1.0), else ``0``.

    Reads the score from the same WebShop environment the tools used (``context.acquire_resource``).
    This is the reward behind verl-agent's reported WebShop success rates
    (``envs.WebshopWorker.step`` binarizes the env score the same way), so it pairs with the
    ``webshop_browser_action`` tool's per-step reward for GiGPO.

    Args:
        context (Context): Injected rollout context; used to acquire the WebShop resource.
        task_id (int): The task's goal (from the data row).

    Returns:
        dict: ``reward`` (0 / 10), ``trajectory/accuracy`` (0 / 1 success, aggregated per
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


# --8<-- [end:webshop_reward_example]


@reward(name="webshop_score_reward")
async def webshop_score_reward(context: Context, task_id: int) -> dict:
    """WebShop dense reward: the environment's purchase score in [0, 1] for the ``task_id`` goal.

    Args:
        context (Context): Injected rollout context; used to acquire the WebShop resource.
        task_id (int): The task's goal (from the data row).

    Returns:
        dict: ``reward`` (the score in [0, 1]), ``trajectory/accuracy`` (1 if the score is 1, else 0)
        and ``output`` (the environment's message). An environment error gives reward 0.
    """
    try:
        env = await context.acquire_resource(spec=WebShopSpec, scope="global", backend="local")
        result = await env.step("get_reward", task_id)
        score = float(result["reward"])
        return {
            # The dense score is the training signal (verl-agent uses it unchanged).
            # ``trajectory/accuracy`` is verl-agent's *success* (score == 1.0), the number
            # their WebShop results report; aggregated per trajectory.
            "reward": score,
            "trajectory/accuracy": float(score == 1.0),
            "output": result["observation"],
        }
    except Exception as e:
        return {
            "reward": 0.0,
            "output": f"Error webshop reward function: {e}",
        }
