from agents.agents.agents.templates.templates import get_conv_template
from agents.agents.templates.utils import compare_hf_template

def test_simple_prompts():
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "Want to play a game?"},
        {"role": "assistant", "content": "Sure, what game?"},
        {"role": "user", "content": "What is 3 times 5?"},
        {"role": "assistant", "content": '''{"name": "multiply", "arguments": {"x": 3, "y": 5}}'''},
        {"role": "tool", "content": "15"},
    ]
    qwen3_model = "Qwen/Qwen3-30B-A3B"
    is_equal, official_prompt, implemented_prompt, highlighted_prompt = compare_hf_template(qwen3_model, "qwen3", messages=messages, add_generation_prompt=True)
    assert is_equal, f"Official prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    print(f"Highlighted prompt:\n\n{highlighted_prompt}")


def test_simple_prompts_with_system():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "Want to play a game?"},
        {"role": "assistant", "content": "Sure, what game?"},
        {"role": "user", "content": "What is 3 times 5?"},
        {"role": "assistant", "content": '''{"name": "multiply", "arguments": {"x": 3, "y": 5}}'''},
        {"role": "tool", "content": "15"},
    ]
    qwen3_model = "Qwen/Qwen3-30B-A3B"
    is_equal, official_prompt, implemented_prompt, highlighted_prompt = compare_hf_template(qwen3_model, "qwen3", messages=messages, add_generation_prompt=True)
    assert is_equal, f"Official prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    print(f"Highlighted prompt:\n\n{highlighted_prompt}")


def test_with_tools():
    multiply_schema = {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}}
    add_schema = {"type": "function", "function": {"name": "add", "description": "A function that add two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "number", "description": "The first number to add"}, "b": {"type": "number", "description": "The second number to add"}}, "required": ["a", "b"]}}}
    tools = [multiply_schema, add_schema]
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "Want to play a game?"},
        {"role": "assistant", "content": "Sure, what game?"},
        {"role": "user", "content": "What is 3 times 5?"},
        {"role": "assistant", "content": '''{"name": "multiply", "arguments": {"x": 3, "y": 5}}'''},
        {"role": "tool", "content": "15"},
    ]
    qwen3_model = "Qwen/Qwen3-30B-A3B"
    is_equal, official_prompt, implemented_prompt, highlighted_prompt = compare_hf_template(qwen3_model, "qwen3", messages=messages, tools=tools, add_generation_prompt=True)
    assert is_equal, f"Official prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    print(f"Highlighted prompt:\n\n{highlighted_prompt}")


def test_with_system_message():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "What is 3 times 5?"},
    ]
    qwen3_model = "Qwen/Qwen3-30B-A3B"
    is_equal, official_prompt, implemented_prompt, highlighted_prompt = compare_hf_template(qwen3_model, "qwen3", messages=messages, enable_thinking=False, add_generation_prompt=True)
    assert is_equal, f"Official prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    print(f"Highlighted prompt:\n\n{highlighted_prompt}")

def test_with_system_message_and_tools():
    multiply_schema = {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}}
    add_schema = {"type": "function", "function": {"name": "add", "description": "A function that add two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "number", "description": "The first number to add"}, "b": {"type": "number", "description": "The second number to add"}}, "required": ["a", "b"]}}}
    tools = [multiply_schema, add_schema]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "What is 3 times 5?"},
    ]
    qwen3_model = "Qwen/Qwen3-30B-A3B"
    is_equal, official_prompt, implemented_prompt, highlighted_prompt = compare_hf_template(qwen3_model, "qwen3", messages=messages, tools=tools, enable_thinking=False,add_generation_prompt=True)
    assert is_equal, f"Official prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    print(f"Highlighted prompt:\n\n{highlighted_prompt}")