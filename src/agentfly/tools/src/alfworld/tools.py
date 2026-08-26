import traceback

from ....core import Context
from ....envs.alfworld_env import ALFWorldSpec
from ...decorator import tool


async def _get_alfworld_env(context: Context):
    """Acquire the ALFWorld resource, resetting it to the episode's task on first use."""
    need_reset = not context.is_spec_acquired(ALFWorldSpec)
    env = await context.acquire_resource(
        spec=ALFWorldSpec,
        scope="global",
        backend="local",
    )
    if need_reset:
        meta = context.metadata or {}
        env_args = {"task_id": meta["task_id"]} if "task_id" in meta else None
        await env.reset(env_args=env_args, split=meta.get("split", "train"))
    return env


def format_observation(observation: str, commands) -> str:
    """Append the admissible-action menu to the observation so the model can pick
    a valid action every turn (mirrors verl-agent's per-step prompt) without
    spending a turn on a separate lookup tool. Shared with the agent's first-node
    hook so the initial reset observation is rendered the same way as step results."""
    text = str(observation)
    if commands:
        text += "\n\nAdmissible actions: [" + ", ".join(commands) + "]"
    return text


@tool(
    name="alfworld_step",
    description="Take an action in the ALFWorld environment and return the resulting observation plus the admissible actions for the new state.",
    stateful=True,
)
async def alfworld_step(action: str, context: Context):
    """
    Take an action in the ALFWorld environment.

    Args:
        action (str): The action to take in the environment.
        context (Context): Injected rollout context; used to acquire the ALFWorld resource.

    Returns:
        dict: ``observation`` (raw text + admissible actions) shown to the model,
        the raw ``anchor`` (undecorated observation) used as the step-grouping key,
        the first-class per-step ``step_reward`` (RL signal), and ``done`` /
        ``invalid_action`` metrics.

    The per-step reward carries an invalid-action penalty (``-0.1`` when the env
    rejects the action, i.e. returns ``"Nothing happens."``), matching verl-agent's
    ALFWorld recipe (``use_invalid_action_penalty``, ``invalid_action_penalty_coef=0.1``).
    ALFWorld's outcome reward is sparse (0 until the winning step), so early on almost
    every anchor group is all-zero-return and the GiGPO step term goes silent; the
    penalty injects reward variance into the large ``"Nothing happens."`` clusters so
    the step advantage can learn to avoid wasted invalid turns. The penalty rides only
    on ``step_reward`` (the GiGPO step signal) — the episode outcome reward comes from
    the separate reward function and is unaffected.
    """
    try:
        env = await _get_alfworld_env(context)
        obs, reward, done, info = await env.step(action)
        commands = info.get("admissible_commands") if info else None
        invalid = str(obs).strip().lower().startswith("nothing happens")
        step_reward = float(reward) - (0.1 if invalid else 0.0)
        return {
            "observation": format_observation(obs, commands),
            "anchor": obs,  # raw state (no admissible-action menu) for GiGPO grouping
            "step_reward": step_reward,
            "metrics": {"done": float(done), "invalid_action": float(invalid)},
        }
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    print("ALFWorld Tools Schema:")
    print("======================")
    print("alfworld_step schema:")
    print(alfworld_step.schema)
