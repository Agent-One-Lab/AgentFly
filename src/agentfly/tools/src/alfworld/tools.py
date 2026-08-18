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
        dict: observation (raw text + admissible actions), reward, done, info.
    """
    try:
        env = await _get_alfworld_env(context)
        obs, reward, done, info = await env.step(action)
        commands = info.get("admissible_commands") if info else None
        return {
            "observation": format_observation(obs, commands),
            "reward": float(reward),
            "done": bool(done),
            "info": info | {"reward": float(reward)},  # keep reward in info
        }
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    print("ALFWorld Tools Schema:")
    print("======================")
    print("alfworld_step schema:")
    print(alfworld_step.schema)
