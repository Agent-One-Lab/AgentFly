from ....templates.utils import tokenize_conversation
import pytest
from transformers import AutoTokenizer
import torch
from ....templates.templates import Chat

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
])
@pytest.mark.parametrize("tools", [
    None,
])
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_template_tokenize(template, messages, tools, add_generation_prompt):
    template_tokenizer_mapping = {
        "qwen2.5": "Qwen/Qwen2.5-3B-Instruct",
        "llama-3.2": "meta-llama/Llama-3.2-3B-Instruct",
        "deepseek-r1-distill-qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    tokenizer = AutoTokenizer.from_pretrained(template_tokenizer_mapping[template], trust_remote_code=True)

    chat = Chat(template, messages, tools=tools)
    prompt = chat.prompt(add_generation_prompt=add_generation_prompt, tools=tools)

    hf_inputs = tokenizer(prompt, return_tensors="pt")

    implemented_inputs = tokenize_conversation(messages, tokenizer, template, max_length=2048, tools=tools, add_generation_prompt=add_generation_prompt, return_tensors="pt")

    assert torch.equal(hf_inputs["input_ids"], implemented_inputs["input_ids"]), f"template: {template}\n\nmessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nprompt: {prompt}\n\nimplemented_prompt: {tokenizer.decode(implemented_inputs['input_ids'][0])}\n\nhf_inputs: {hf_inputs}\n\nimplemented_inputs: {implemented_inputs}"