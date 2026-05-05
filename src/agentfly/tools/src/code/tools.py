import traceback

from ....core import Context
from ....envs.python_env import PythonSandboxSpec
from ...decorator import tool
from ...tool_base import BaseTool


# --8<-- [start:code_interpreter_example]
@tool(
    name="code_interpreter",
    description="Run the code in docker container and return the output from stdout or stderr",
)
async def code_interpreter(code: str, context: Context):
    """
    Run the code in docker container and return the output from stdout or stderr.

    Uses one Python sandbox per rollout (acquired via Context); the sandbox is
    released when the rollout ends. Warm the pool at training start with
    ResourceEngine.start(python_sandbox_spec(), size=32, backend="local") if needed.

    Args:
        code: The code to run.
        context: Injected rollout context; used to acquire the sandbox resource.

    Returns:
        str: The output from stdout or stderr.
    """
    code = str(code)
    spec = PythonSandboxSpec
    env = await context.acquire_resource(spec=spec, scope="global", backend="local")
    try:
        obs = await env.step(code)
        return str(obs)
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"
# --8<-- [end:code_interpreter_example]


class CodeInterpreterTool(BaseTool):
    name = "code_interpreter_tool"
    description = (
        "Run the code in docker container and return the output from stdout or stderr"
    )

    def __init__(self):
        super().__init__()

    async def call(self, code: str, context: Context):
        """
        Args:
            code: The code to run.
            context: Injected rollout context for acquiring the sandbox.
        """
        code = str(code)
        spec = PythonSandboxSpec
        env = await context.acquire_resource(spec=spec, scope="global", backend="local")
        try:
            obs = await env.step(code)
            return str(obs)
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    print(code_interpreter.schema)
