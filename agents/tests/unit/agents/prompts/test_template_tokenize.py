from agents.agents.templates.utils import is_vlm_template, tokenize_conversation
import pytest
from transformers import AutoTokenizer, AutoProcessor
import torch
from agents.agents.templates.templates import Chat

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
    # [
    #     {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
    #     {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
    # ]
])
@pytest.mark.parametrize("add_generation_prompt", [False])
def test_template_tokenize(template, messages, tools, add_generation_prompt):
    template_tokenizer_mapping = {
        "qwen2.5": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-vl": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen3": "Qwen/Qwen3-8B",
    }
    tokenizer = AutoTokenizer.from_pretrained(template_tokenizer_mapping[template])
    if is_vlm_template(template):
        processor = AutoProcessor.from_pretrained(template_tokenizer_mapping[template])
    else:
        processor = None
    try:
        official_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        official_inputs = tokenizer(official_prompt, return_tensors="pt")
        # chat = Chat(template, messages, tokenizer)
        # implemented_inputs = chat.tokenize()
        implemented_inputs = tokenize_conversation(messages, tokenizer, template, max_length=2048, processor=processor, tools=tools, add_generation_prompt=add_generation_prompt, return_tensors="pt")
        
        assert torch.equal(official_inputs["input_ids"], implemented_inputs["input_ids"]), f"template: {template}\n\nmessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nofficial_prompt: {official_prompt}\n\nimplemented_prompt: {tokenizer.decode(implemented_inputs['input_ids'][0])}\n\nofficial_inputs: {official_inputs}\n\nimplemented_inputs: {implemented_inputs}"
        assert torch.equal(official_inputs["attention_mask"], implemented_inputs["attention_mask"])
    except Exception as e:
        if isinstance(e, ValueError) and "does not support tool calling." in str(e) and tools is not None and template in ["qwen2.5-vl"]:
            pass
        else:
            raise e
    # assert torch.equal(official_inputs["labels"], implemented_inputs["labels"])
    
    
