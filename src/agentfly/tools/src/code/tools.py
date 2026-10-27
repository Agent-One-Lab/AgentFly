"""Code tools: run Python in a sandbox container.

``code_interpreter`` (decorator-based) and :class:`CodeInterpreterTool` (class-based) execute code
in a Python sandbox acquired through the context (``scope="global"``) and return stdout/stderr.
"""

import traceback

from ....core import Context
from ....envs.python_env import PythonSandboxSpec
from ...decorator import tool
from ...tool_base import BaseTool


# --8<-- [start:code_interpreter_example]
@tool(
    name="code_interpreter",
    description=(
        "Run Python code in a sandbox and return its stdout or stderr. Each call runs in a fresh "
        "Python process, so define everything the code needs (imports, variables, functions) in the same call."
    ),
)
async def code_interpreter(code: str, context: Context):
    """
    Run Python code in a sandbox and return its stdout or stderr.

    The sandbox container comes from a pool shared by all rollouts (``scope="global"``) and
    goes back to it when the rollout ends; every call runs in a fresh Python process, so no
    state leaks between calls or rollouts. Warm the pool at training start with
    ``ResourceEngine.start(PythonSandboxSpec, size=32, backend="local")`` if needed.

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
        "Run Python code in a sandbox and return its stdout or stderr. Each call runs in a fresh "
        "Python process, so define everything the code needs (imports, variables, functions) in the same call."
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
