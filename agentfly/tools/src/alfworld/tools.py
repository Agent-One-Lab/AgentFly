import asyncio
import traceback

from ....envs.alfworld_env import ALFWorldEnv
from ...decorator import tool


@tool(env_cls=ALFWorldEnv, name="alfworld_step", description="Take an action in the ALFWorld environment and return the observation", stateful=True, pool_size=8)
async def alfworld_step(action: str, env: ALFWorldEnv):
    try:
        obs, reward, done, info = await env.step(action)
        return {
        "observation": obs,
        "reward": float(reward),
        "done": bool(done),
        "info": info | {"reward": float(reward)}   # keep reward in info
    }
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@tool(env_cls=ALFWorldEnv, name="alfworld_reset", description="Reset the ALFWorld environment to start a new episode", stateful=True, pool_size=32)
async def alfworld_reset(env: ALFWorldEnv):
    try:
        obs, info = await env.reset()
        return obs
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@tool(env_cls=ALFWorldEnv, name="alfworld_get_admissible_commands", description="Get the list of admissible commands for the current state in ALFWorld", stateful=True, pool_size=8)
async def alfworld_get_admissible_commands(env: ALFWorldEnv):
    try:
        commands = await env.get_admissible_commands()
        return "\n".join(commands)
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"
    
@tool(env_cls=ALFWorldEnv, name="alfworld_get_task_objective",
      description="Get the current task objective/goal from the ALFWorld environment",
      stateful=True, pool_size=8)
async def alfworld_get_task_objective(env: ALFWorldEnv):
    try:
        # First ask the environment. If it returns nothing, fall back to the info
        # cached during the last reset/step (env._current_info).
        info = await env.get_info()
        if not info:                      # HTTP endpoint returned nothing
            info = getattr(env, "_current_info", {}) or {}

        task_objective = info.get(
            "goal",
            info.get("task_description",
                     info.get("task", "No task objective found"))
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