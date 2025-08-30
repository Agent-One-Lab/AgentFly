import pytest
from ....tools.tool_base import tool, submit_tool_calls

# def test_tool_call_sync():
#     # Create a custom sync tool that doesn't use the async implementation
#     @tool(name="add_numbers_sync", description="Add two numbers")
#     def add_numbers(a: int, b: int):
#         return a + b
    
#     # Test non-stateful tool
#     result1 = add_numbers.call(a=2, b=3)
#     assert result1["observation"] == '5', f"Expected 5 but got {result1['observation']}"
    
#     # Test with the submit_tool_calls function
#     tool_names = ["add_numbers_sync", "code_interpreter"]
#     tool_inputs = [{"a": 2, "b": 3}, {"code": "print('Hello, world!')"}]
#     ids = [None, "test_tool_id1"]
#     results = submit_tool_calls(tool_names, tool_inputs, ids)
#     assert results[0]["observation"] == '5', f"Expected 5 but got {results[0]['observation']}"
    # assert results[1]["observation"] == "Hello, world!\n", f"Expected 'Hello, world!\n' but got {results[1]['observation']}"