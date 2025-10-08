from ....templates.utils import compare_hf_template
import pytest
from transformers import AutoTokenizer

@pytest.mark.parametrize("template", ["qwen3"])
@pytest.mark.parametrize("messages", [
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "<think> This is test thinking content. </think> I am fine, thank you."},
        {"role": "user", "content": "Want to play a game?"},
        {"role": "assistant", "content": "<think> This is test thinking content. </think> Sure, what game?"},
    ],
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "<think> This is test thinking content. </think> I am fine, thank you."},
    ],
    [
        {"role": "user", "content": "Help me to calculate 3 times 5."},
        {"role": "assistant", "content": '''{"name": "multiply", "arguments": {"x": 3, "y": 5}}'''},
        {"role": "tool", "content": "15"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "<think> This is test thinking content. </think> I am fine, thank you."},
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
@pytest.mark.parametrize("enable_thinking", [True, False])
def test_chat_template_equal(template, messages, tools, add_generation_prompt, enable_thinking):
    # Filter invalid combinations
    if add_generation_prompt and messages[-1]['role'] == 'assistant':
        return
    
    template_tokenizer_mapping = {
        "qwen3": "Qwen/Qwen3-32B",
    }
    tokenizer = AutoTokenizer.from_pretrained(template_tokenizer_mapping[template], trust_remote_code=True)

    is_equal, is_equal_between_implemented_prompts, is_equal_between_jinja_prompts, official_prompt, implemented_prompt, implemented_jinja_prompt, highlighted_prompt = compare_hf_template(tokenizer, template, messages=messages, tools=tools,add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
    
    print(f"Official prompt:\n\n{official_prompt}")
    print(f"Implemented prompt:\n\n{implemented_prompt}")
    print(f"Highlighted prompt:\n\n{highlighted_prompt}")
    assert is_equal, f"Template: {template}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nenable_thinking: {enable_thinking}\n\nOfficial prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    assert is_equal_between_jinja_prompts, f"Template: {template}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}\n\nJinja prompt:\n\n{implemented_jinja_prompt}"
    # print(f"Official prompt:\n\n{official_prompt}")
    # print(f"Highlighted prompt:\n\n{highlighted_prompt}")


