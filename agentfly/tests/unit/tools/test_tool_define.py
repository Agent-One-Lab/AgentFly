from ....tools.tool_base import tool
from ....envs.python_env import PythonSandboxEnv
import pytest

def test_base_tool():
    @tool(name="test_tool", description="test tool")
    def test_tool(name="test_tool"):
        return "test"
    
    assert test_tool.name == "test_tool"
    print(test_tool.schema)

# @pytest.mark.asyncio(loop_scope="session")
# async def test_stateful_tool():
#     @tool(env_cls=PythonSandboxEnv, name="test_tool", description="test tool", stateful=True)
#     async def test_tool(code: str):
#         env = current_env.get()
#         obs = await env.step(code)
#         return obs
    
#     assert test_tool.name == "test_tool"
#     print(test_tool.schema)

#     result = await test_tool(code="print('Hello, world!')", id="test_tool_id0")
#     assert result['observation'] == "Hello, world!\n", f"{result}"
