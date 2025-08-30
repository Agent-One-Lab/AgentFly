import pytest
from ....tools import code_interpreter
from ....tools.tool_base import tool, Tool
from ....envs.python_env import PythonSandboxEnv


# def test_stateful_tool_sync():
#     """Test a stateful tool (code_interpreter)"""
#     # Use the synchronous call method, providing a required ID parameter
#     result = code_interpreter.call(code="print('Hello, world!')", id="test_id")
    
#     assert result["observation"] == "Hello, world!\n", f"Expected 'Hello, world!\n' but got {result['observation']}"


# def test_nonstateful_tool_sync():
#     """Test a non-stateful tool"""
#     # Create a simple non-stateful tool
#     @tool(name="add_numbers", description="Add two numbers")
#     def add_numbers(a: int, b: int):
#         return a + b
    
#     # Call it synchronously without an ID
#     result = add_numbers.call(a=2, b=3)
    
#     assert result["observation"] == '5', f"Expected 5 but got {result['observation']}"


# def test_direct_tool_creation_sync():
#     """Test creating a Tool directly without the decorator"""
#     # Create a tool directly
#     def multiply(a: int, b: int):
#         return a * b
        
#     multiply_tool = Tool(
#         func=multiply,
#         name="multiply_numbers",
#         description="Multiply two numbers",
#         stateful=False
#     )
    
#     # Call it synchronously
#     result = multiply_tool.call(a=3, b=4)
    
#     assert result["observation"] == '12', f"Expected 12 but got {result['observation']}"

