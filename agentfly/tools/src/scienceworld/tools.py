import traceback

from ...tool_base import tool

from ....envs.scienceworld_env import ScienceWorldEnv

@tool(env_cls=ScienceWorldEnv, name="scienceworld_explorer", description="Take an action in scienceworld environment and return the observation. Valid actions are 'look around', 'inventory', 'task', 'open <OBJ>', 'close <OBJ>', 'deactivate <OBJ>', 'activate <OBJ>', 'connect <OBJ> to <OBJ2>', 'disconnect <OBJ>', 'use <OBJ>', 'use <OBJ> [on <OBJ2>]' (OBJ2 is optional), 'look at <OBJ>', 'look in <OBJ>', 'read <OBJ>', 'move <OBJ> to <OBJ2>', 'pick up <OBJ>', 'put down <OBJ>', 'pour <OBJ> into <OBJ2>', 'dunk <OBJ> into <OBJ2>', 'mix <OBJ>', 'go to <LOC>', 'eat <OBJ>', 'flush <OBJ>', 'focus on <OBJ>', 'wait [<DURATION>]' (DURATION is optional)",  stateful=True, pool_size=8)
async def scienceworld_explorer(action: str, env: ScienceWorldEnv):
    """
    Executes an action in the ScienceWorld environment and returns the resulting observation.

    Parameters:
        action (str): The action to perform in the environment. Valid actions include commands like 'look around', 'inventory', 'open <OBJ>', etc.
        env (ScienceWorldEnv): The ScienceWorld environment instance in which the action will be performed.

    Returns:
        str: The observation returned by the environment after performing the action, or an error message if the action is invalid or an exception occurs.
    """
    try:
        observation = await env.step(action)
        return observation
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    print(scienceworld_explorer.schema)