import pytest
from transformers import AutoTokenizer
from ....templates.templates import get_template
from ....templates.utils import compare_hf_template

# nemotron, phi-4, glm-4
@pytest.mark.parametrize("template_name", ["qwen2.5-think", "qwen2.5-no-system-tool",])
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
        {"role": "tool", "content": "15", "tool_call_id": "123456789"},
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
def test_chat_template_equal(template_name, messages, tools, add_generation_prompt):
    # Filter invalid combinations
    if add_generation_prompt and messages[-1]['role'] == 'assistant':
        return

    template = get_template(template_name)
    if tools and not template._supports_tool_call():
        return
    
    contain_tool_role = any(message['role'] == 'tool' for message in messages)
    if contain_tool_role and not template._supports_tool_call():
        return
    
    template_tokenizer_mapping = {
        "qwen2.5": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-think": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-no-system-tool": "Qwen/Qwen2.5-3B-Instruct",
        "deepseek-prover-v2": "deepseek-ai/DeepSeek-Prover-V2-7B",
        "llama-3.2": "meta-llama/Llama-3.2-3B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "glm-4": "THUDM/glm-4-9b-chat",
        "internlm2.5": "internlm/internlm2_5-7b-chat",
        "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
        "phi-4": "microsoft/Phi-4",
        "nemotron": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    }
    tokenizer = AutoTokenizer.from_pretrained(template_tokenizer_mapping[template_name], trust_remote_code=True)

    is_equal, is_equal_between_implemented_prompts, is_equal_between_jinja_prompts, official_prompt, implemented_prompt, implemented_jinja_prompt, highlighted_prompt = compare_hf_template(tokenizer, template_name, messages=messages, tools=tools,add_generation_prompt=add_generation_prompt)
    
    print(f"Official prompt:\n\n\"{official_prompt}\"\n\n")
    print(f"Implemented prompt:\n\n\"{implemented_prompt}\"\n\n")
    assert is_equal, f"Template: {template}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nOfficial prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    assert is_equal_between_jinja_prompts, f"Template: {template}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}\n\nJinja prompt:\n\n{implemented_jinja_prompt}"