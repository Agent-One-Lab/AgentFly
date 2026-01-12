import asyncio
import json
import requests
import traceback

from ....envs.python_env import PythonSandboxEnv
from ...decorator import tool
from ...tool_base import BaseTool

def make_request(url, payload, headers, timeout=20):
    """Make a single request to the server"""
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
            verify=False
        )
        response.raise_for_status()  # This will raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out after {} seconds".format(timeout)}
    except requests.exceptions.ConnectionError as e:
        return {"error": f"Connection error: {str(e)}\nPlease check if the server is running and accessible."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}\n{traceback.format_exc()}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from server"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}\n{traceback.format_exc()}"}

    
@tool(env_cls=PythonSandboxEnv, name="code_interpreter", description="Run the code in docker container and return the output from stdout or stderr", stateful=True, pool_size=32)
async def code_interpreter(code: str, env: PythonSandboxEnv):
    """
    Run the code in docker container and return the output from stdout or stderr
    
    Args:
        code (str): The code to run.
        env (PythonSandboxEnv): The Python sandbox environment instance
    
    Returns:
        str: The output from stdout or stderr
    """
    code = str(code)
    try:
        obs = await env.step(code)
        return str(obs)
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


class CodeInterpreterTool(BaseTool):
    name = "code_interpreter_tool"
    description = "Run the code in docker container and return the output from stdout or stderr"
    env_cls = PythonSandboxEnv
    pool_size = 32
    def __init__(self):
        super().__init__()

    async def call(self, code: str, env: PythonSandboxEnv):
        code = str(code)
        try:
            obs = await env.step(code)
            return str(obs)
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"

if __name__ == "__main__":
    print(code_interpreter.schema)