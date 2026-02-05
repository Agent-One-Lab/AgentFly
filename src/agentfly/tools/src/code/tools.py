import traceback

from ....envs.python_env import python_sandbox_spec
from ...decorator import tool
from ...tool_base import BaseTool


@tool(
    resource_spec=python_sandbox_spec(),
    name="code_interpreter",
    description="Run the code in docker container and return the output from stdout or stderr",
    stateful=True,
    pool_size=32,
    backend="local",
)
async def code_interpreter(code: str, resource):
    """
    Run the code in docker container and return the output from stdout or stderr

    Args:
        code (str): The code to run.
        resource: The Python sandbox resource (injected by BaseTool).

    Returns:
        str: The output from stdout or stderr
    """
    code = str(code)
    try:
        obs = await resource.step(code)
        return str(obs)
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


class CodeInterpreterTool(BaseTool):
    name = "code_interpreter_tool"
    description = (
        "Run the code in docker container and return the output from stdout or stderr"
    )
    resource_spec = python_sandbox_spec()
    backend = "local"
    pool_size = 32

    def __init__(self):
        super().__init__()

    async def call(self, code: str, resource):
        code = str(code)
        try:
            obs = await resource.run_code(code)
            return str(obs)
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    print(code_interpreter.schema)
