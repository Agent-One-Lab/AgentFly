""" This file is for testing the tokenization of the templates. The templates should align on following aspects:
    - The tokenized prompt should be the same as the one obtained from HF template with all the following options:
        - add_generation_prompt
        - tools
    - We need to observe the labels and action_mask to make sure the the they are correct.

Since the align for textual prompt is already tested in other files, we only need to test the tokenization of the templates.
"""

from ....templates.utils import tokenize_conversation
import pytest
from transformers import AutoTokenizer
import torch
from ....templates.templates import Chat

@pytest.mark.parametrize("template", ["llama-3.2", "qwen2.5"])
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
        {"role": "assistant", "content": "15"},
        {"role": "user", "content": "OK, what is 3 times 6?"},
        {"role": "assistant", "content": "18"},
    ],
])
@pytest.mark.parametrize("tools", [
    None,
    [
        {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
        {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
    ]
])
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_template_tokenize(template, messages, tools, add_generation_prompt):
    template_tokenizer_mapping = {
        "qwen2.5": "Qwen/Qwen2.5-3B-Instruct",
        "llama-3.2": "meta-llama/Llama-3.2-3B-Instruct",
    }
    tokenizer = AutoTokenizer.from_pretrained(template_tokenizer_mapping[template], trust_remote_code=True)

    chat = Chat(template, messages, tools=tools)
    prompt = chat.prompt(add_generation_prompt=add_generation_prompt, tools=tools)

    hf_inputs = tokenizer(prompt, return_tensors="pt")

    implemented_inputs = tokenize_conversation(messages, tokenizer, template, max_length=2048, tools=tools, add_generation_prompt=add_generation_prompt, return_tensors="pt")

    assert torch.equal(hf_inputs["input_ids"], implemented_inputs["input_ids"]), f"template: {template}\n\nmessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nprompt: {prompt}\n\nimplemented_prompt: {tokenizer.decode(implemented_inputs['input_ids'][0])}\n\nhf_inputs: {hf_inputs}\n\nimplemented_inputs: {implemented_inputs}"
