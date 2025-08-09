""" This file is for testing the text templates that align seamlessly with HF templates. The templates should align on following aspects:
    - The obtained textual prompt should be the same as the one obtained from HF template with all the following options:
        - add_generation_prompt
        - tools
    - The obtained textual prompt should be the same as the one obtained from Jinja template with all the following options:
        - add_generation_prompt
        - tools
"""


from agents.agents.templates.utils import compare_hf_template
from transformers import AutoTokenizer
import pytest

@pytest.mark.parametrize("model_name_or_path", ["Qwen/Qwen2.5-3B-Instruct"])
@pytest.mark.parametrize("messages", [
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "Want to play a game?"},
        {"role": "assistant", "content": "Sure, what game?"},
    ],
    [
        {"role": "user", "content": "Help me to calculate 3 times 5."},
        {"role": "assistant", "content": '''{"name": "multiply", "arguments": {"x": 3, "y": 5}}'''},
        {"role": "tool", "content": "15"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "What is 3 times 5?"},
    ],
])
@pytest.mark.parametrize("tools", [
    None,
    [
        {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
        {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
    ]
])
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_hf_template_print(model_name_or_path, messages, tools, add_generation_prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, tools=tools)
    print(f"========================================\nModel: {model_name_or_path}\nMessages: {messages}\nTools: {tools}\nAdd generation prompt: {add_generation_prompt}\n")
    print(prompt)
    print("========================================\n")


# "qwen2.5-think", "qwen2.5", "qwen2.5-no-tool",
@pytest.mark.parametrize("template", ["qwen2.5"])
@pytest.mark.parametrize("messages", [
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "Want to play a game?"},
        {"role": "assistant", "content": "Sure, what game?"},
    ],
    [
        {"role": "user", "content": "Help me to calculate 3 times 5."},
        {"role": "assistant", "content": '''{"name": "multiply", "arguments": {"x": 3, "y": 5}}'''},
        {"role": "tool", "content": "15"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "What is 3 times 5?"},
    ],
])
@pytest.mark.parametrize("tools", [
    None,
    [
        {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
        {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
    ]
])
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_chat_template_equal(template, messages, tools, add_generation_prompt):
    # Filter invalid combinations
    if add_generation_prompt and messages[-1]['role'] == 'assistant':
        return
    
    template_tokenizer_mapping = {
        "qwen2.5": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-think": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-no-system-tool": "Qwen/Qwen2.5-3B-Instruct",
        "deepseek-prover-v2": "deepseek-ai/DeepSeek-Prover-V2-7B",
    }
    tokenizer = AutoTokenizer.from_pretrained(template_tokenizer_mapping[template])

    is_equal, is_equal_between_implemented_prompts, is_equal_between_jinja_prompts, official_prompt, implemented_prompt, implemented_jinja_prompt, highlighted_prompt = compare_hf_template(tokenizer, template, messages=messages, tools=tools,add_generation_prompt=add_generation_prompt)
    assert is_equal, f"Template: {template}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nOfficial prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    assert is_equal_between_jinja_prompts, f"Template: {template}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}\n\nJinja prompt:\n\n{implemented_jinja_prompt}"
    print(f"Official prompt:\n\n{official_prompt}")
    print(f"Highlighted prompt:\n\n{highlighted_prompt}")

