import traceback

from ....core import Context
from ....envs.alfworld_env import ALFWorldSpec
from ...decorator import tool


async def _get_alfworld_env(context: Context):
    return await context.acquire_resource(spec=ALFWorldSpec, scope="global", backend="local")


@tool(
    name="alfworld_step",
    description="Take an action in the ALFWorld environment and return the observation",
    stateful=True,
)
async def alfworld_step(action: str, context: Context):
    """
    Take an action in the ALFWorld environment and return the observation

    Args:
        action (str): The action to take in the environment
        context (Context): Injected rollout context; used to acquire the ALFWorld resource.

    Returns:
        dict: A dictionary containing the observation, reward, done, and info
    """
    try:
        env = await _get_alfworld_env(context)
        obs, reward, done, info = await env.step(action)
        return {
            "observation": obs,
            "reward": float(reward),
            "done": bool(done),
            "info": info | {"reward": float(reward)},  # keep reward in info
        }
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="alfworld_reset",
    description="Reset the ALFWorld environment to start a new episode",
    stateful=True,
)
async def alfworld_reset(context: Context):
    try:
        env = await _get_alfworld_env(context)
        obs, info = await env.reset()
        return obs
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="alfworld_get_admissible_commands",
    description="Get the list of admissible commands for the current state in ALFWorld",
    stateful=True,
)
async def alfworld_get_admissible_commands(context: Context):
    try:
        env = await _get_alfworld_env(context)
        commands = await env.get_admissible_commands()
        return "\n".join(commands)
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="alfworld_get_task_objective",
    description="Get the current task objective/goal from the ALFWorld environment",
    stateful=True,
)
async def alfworld_get_task_objective(context: Context):
    try:
        env = await _get_alfworld_env(context)
        info = await env.get_info()
        if not info:
            info = getattr(env, "_current_info", {}) or {}

        task_objective = info.get(
            "goal",
            info.get("task_description", info.get("task", "No task objective found")),
        )
        task_type = info.get("task_type", "Unknown task type")

        return f"Task: {task_objective}\nTask Type: {task_type}"
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    print("ALFWorld Tools Schema:")
    print("======================")
    print("alfworld_step schema:")
    print(alfworld_step.schema)
    print("\nalfworld_get_admissible_commands schema:")
    print(alfworld_get_admissible_commands.schema)
