from ....templates import compare_hf_template
from transformers import AutoTokenizer
import pytest


# "qwen2.5-think", "qwen2.5", "qwen2.5-no-tool",
# "llama-3.2", "mistral", "glm-4", "internlm2.5", "phi-3.5", "phi-4"
@pytest.mark.parametrize("template", ["deepseek-r1-distill-qwen"])
@pytest.mark.parametrize("messages", [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
    ],
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "<think> This is test thinking content. </think> I am fine, thank you."},
    ],
    [
        {"role": "user", "content": "Hello, how are you?"},
    ],
])
@pytest.mark.parametrize("tools", [
    None,
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
        "llama-3.2": "meta-llama/Llama-3.2-3B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "glm-4": "THUDM/glm-4-9b-chat",
        "internlm2.5": "internlm/internlm2_5-7b-chat",
        "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
        "phi-4": "microsoft/Phi-4",
        "nemotron": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "deepseek-r1-distill-qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    tokenizer = AutoTokenizer.from_pretrained(template_tokenizer_mapping[template], trust_remote_code=True)

    is_equal, is_equal_between_implemented_prompts, is_equal_between_jinja_prompts, official_prompt, implemented_prompt, implemented_jinja_prompt, highlighted_prompt = compare_hf_template(tokenizer, template, messages=messages, tools=tools,add_generation_prompt=add_generation_prompt)
    
    print(f"Official prompt:\n\n{official_prompt}")
    print(f"Highlighted prompt:\n\n{highlighted_prompt}")
    assert is_equal, f"Template: {template}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nOfficial prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    assert is_equal_between_jinja_prompts, f"Template: {template}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}\n\nJinja prompt:\n\n{implemented_jinja_prompt}"
