from ....tools import submit_tool_call, tool, submit_tool_calls
import pytest
import asyncio

# @pytest.mark.asyncio(loop_scope="session")
# async def test_tool_call_by_name_predefined():
#     tool_name = "code_interpreter"
#     tool_input = {
#         "code": "print('Hello, world!')"
#     }
#     result = await submit_tool_call(tool_name, tool_input, "test_tool_id0")
#     assert result['observation'] == "Hello, world!\n", f"{result}"


# @pytest.mark.asyncio(loop_scope="session")
# async def test_tool_call_by_name_custom():
#     tool_name = "add_numbers"
#     @tool(name=tool_name, description="Add two numbers")
#     def add_numbers(a: int, b: int):
#         return a + b
    
#     result = await submit_tool_call(tool_name, {"a": 2, "b": 3})
    
#     assert result["observation"] == '5', f"Expected 5 but got {result['observation']}"



