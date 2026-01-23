from agentfly.tools import CodeInterpreterTool
import pytest


def test_code_schema():
    instance_schema = CodeInterpreterTool().schema
    print(instance_schema)


@pytest.mark.asyncio(loop_scope="session")
async def test_code_run():
    tool_instance = CodeInterpreterTool()
    print(f"Stateful of {tool_instance}: {tool_instance.is_stateful}")
    result = await tool_instance(code="print('Hello, World!')")
    print(result)
