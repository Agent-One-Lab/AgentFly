import traceback
from typing import Any, Dict, List, Optional

from ....core import Context
from ....envs.webshop_text_env import WebShopSpec
from ...decorator import tool


@tool(
    name="webshop_browser",
    description="Browse the webshop by searching or clicking. The action is either 'search' or 'click' and the value is the search query or the element to click. Clickables: 'Buy Now', 'Next >', '< Prev', 'Back to Search', 'Description', 'Features', 'Reviews', 'Attributes', product ASIN or ID like 'B079HGJ5MH' and their attributes or variants like 'Yellow', 'Blue', 'Small', 'Large', 'XL', '40x60', etc.",
    stateful=True,
)
async def webshop_browser(action: str, value: str, context: Context):
    """
    Interact with the webshop environment by performing a search or clicking an element.

    Args:
        action (str): The action to perform, either 'search' or 'click'.
        value (str): The search query or the element to click (e.g., button, product ID, attribute).
        context (Context): Injected rollout context; used to acquire the WebShop resource.

    Returns:
        str: The observation from the environment after performing the action, or an error message if the action is invalid or an exception occurs.
    """
    try:
        need_reset = not context.is_spec_acquired(WebShopSpec)
        env = await context.acquire_resource(
            spec=WebShopSpec,
            scope="global",
            backend="local",
        )
        if need_reset:
            await env.reset(env_args=context.metadata)
        if action == "search":
            observation = await env.step(f"search[{value}]")
        elif action == "click":
            observation = await env.step(f"click[{value}]")
        else:
            return (
                f"Error: Invalid action '{action}'. Must be either 'search' or 'click'"
            )
        return observation
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


# verl-agent's WebShop reward (envs.WebshopWorker.step): 10 on a successful purchase
# (score == 1.0), 0 otherwise. The dense score is not a training signal there.
WEBSHOP_WON_REWARD = 10.0


# ---- verl-agent-aligned page / action formatting (env_manager.WebshopEnvironmentManager) ----

def extract_task(obs: str) -> Optional[str]:
    """verl-agent's ``extract_task``: the ``[SEP]`` part right after ``Instruction:``, or None.

    Every WebShop page repeats the instruction at the top; the title part before it is
    present on the search page only, so the marker is located rather than assumed at
    index 1."""
    parts = str(obs).split(" [SEP] ")
    if "Instruction:" in parts:
        i = parts.index("Instruction:")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def format_page(obs: str, task: Optional[str]) -> str:
    """verl-agent's ``format_obs``: drop everything up to and including the instruction in the
    ``[SEP]``-joined page and quote the remaining parts. The instruction is located by its
    ``Instruction:`` marker (falling back to the task text); a page without it (e.g. the
    ``done`` page) is returned as-is."""
    parts = str(obs).split(" [SEP] ")
    if "Instruction:" in parts:
        index = parts.index("Instruction:") + 1
    elif task is not None and task in parts:
        index = parts.index(task)
    else:
        return str(obs)
    return " [SEP] ".join(f"'{p}'" for p in parts[index + 1:])


def format_avail_actions(avail: Dict[str, Any]) -> List[str]:
    """verl-agent's ``format_avail_actions``: ``search[<your query>]`` when the page has a
    search bar, then one ``click[<clickable>]`` per clickable, in page order."""
    actions: List[str] = []
    if avail.get("has_search_bar"):
        actions.append("search[<your query>]")
    for txt in avail.get("clickables") or []:
        actions.append(f"click[{txt}]")
    return actions


def format_observation(page: str, actions: List[str]) -> str:
    """The LLM-facing observation: the page plus the admissible-action menu rendered the
    way verl-agent's prompt lists it (one quoted action per line). Shared with the
    agent's first-step hook so the initial page renders like every later step."""
    text = str(page)
    if actions:
        text += "\n\nAdmissible actions:\n[\n" + "\n".join(f"'{a}'," for a in actions) + "\n]"
    return text


def _is_rejected(result: Any) -> bool:
    """The env answers a rejected action with a message and leaves the page unchanged."""
    return isinstance(result, str) and (
        result.startswith("Invalid action") or result.startswith("You are not in the search page")
    )


@tool(
    name="webshop_browser_action",
    description="Browse the webshop by searching or clicking. The action is either 'search[<query>]' or 'click[<element>]' and must be one of the admissible actions shown with the current page.",
    stateful=True,
    pool_size=16,
)
async def webshop_browser_action(action: str, context: Context):
    """
    Take one action in the WebShop environment (``search[...]`` / ``click[...]``).

    Args:
        action (str): ``search[<query>]`` or ``click[<clickable>]``.
        context (Context): Injected rollout context; used to acquire the WebShop resource.

    Returns:
        dict: ``observation`` (the page + admissible-action menu) shown to the model, the
        undecorated ``anchor`` page used as the step-grouping key, the per-step
        ``step_reward`` (verl-agent's success reward: 10 on a Buy whose score is exactly 1.0,
        else 0), ``control="end"`` on the Buy step, and ``done`` / ``won`` / ``invalid_action`` /
        ``task_score`` (the dense WebShop score in [0, 1], a diagnostic) metrics
        plus the raw ``admissible_actions`` list for the flat prompt builder.

    Mirrors ``alfworld_step``: the invalid-action penalty is NOT folded into
    ``step_reward`` — the GiGPO estimator applies it post-discount from the
    ``invalid_action`` flag. ``won`` is verl-agent's success definition (score == 1.0).
    """
    # verl-agent's webshop_projection lowercases the extracted action before stepping
    # (``extracted_action.strip().lower()``); the env matches clickables in lowercase, so a
    # capitalized ``Click[...]`` / ``Search[...]`` would otherwise read as an invalid action.
    action = action.strip().lower()
    if action.startswith("choose"):
        action = "click" + action[6:]
    try:
        need_reset = not context.is_spec_acquired(WebShopSpec)
        env = await context.acquire_resource(
            spec=WebShopSpec,
            scope="global",
            backend="local",
        )
        if need_reset:
            await env.reset(env_args=context.metadata)
        result = await env.step(action)
        invalid = _is_rejected(result)
        raw = env.observation  # current page ([SEP]-joined text); unchanged when rejected
        done = "done" in str((env.state or {}).get("url") or "")
        if done:
            # The Buy page. Fetch the dense score for THIS task (the env's /done endpoint).
            score_result = await env.step("get_reward", (context.metadata or {}).get("task_id"))
            score = float(score_result.get("reward", 0.0)) if isinstance(score_result, dict) else 0.0
            page, actions = str(raw), []
        else:
            score = 0.0
            task = extract_task(raw) or env.get_instruction_text()
            page = format_page(raw, task)
            actions = format_avail_actions(env.get_available_actions())
        return {
            "observation": format_observation(page, actions),
            "anchor": page,
            "control": "end" if done else None,
            "step_reward": WEBSHOP_WON_REWARD * float(done and score == 1.0),
            "metrics": {
                "done": float(done),
                "won": float(done and score == 1.0),
                "task_score": score,
                "invalid_action": float(invalid),
                "admissible_actions": actions,
            },
        }
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    print(webshop_browser.schema)
