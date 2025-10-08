""" This file is for testing the text templates that align seamlessly with HF templates. The templates should align on following aspects:
    - The obtained textual prompt should be the same as the one obtained from HF template with all the following options:
        - add_generation_prompt
        - tools
    - The obtained textual prompt should be the same as the one obtained from Jinja template with all the following options:
        - add_generation_prompt
        - tools
"""


from ....templates.templates import Chat
from ....templates.utils import compare_hf_template, tokenize_conversation
from transformers import AutoTokenizer
import pytest
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

@pytest.mark.parametrize("template", ["qwen2.5-vl"])
@pytest.mark.parametrize("messages", [
    [
        {
            "role": "system",
            "content": "You are a multi-modal assistant that can answer questions about images.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The image is a cat.",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in the image?",
                },
            ],
        }
    ],
    # [
    #     {"role": "user", "content": "Help me to calculate 3 times 5."},
    #     {"role": "assistant", "content": '''{"name": "multiply", "arguments": {"x": 3, "y": 5}}'''},
    #     {"role": "tool", "content": "15"},
    # ],
    # [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Hello, how are you?"},
    #     {"role": "assistant", "content": "I am fine, thank you."},
    #     {"role": "user", "content": "What is 3 times 5?"},
    # ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type": "text", "text": "Describe these images."},
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The image is a cat.",
                },
            ],
        },
    ]
])
@pytest.mark.parametrize("tools", [
    None,
    # [
    #     {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
    #     {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
    # ]
])
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_chat_template_equal(template, messages, tools, add_generation_prompt):
    template_tokenizer_mapping = {
        "qwen2.5-vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    }
    tokenizer = AutoTokenizer.from_pretrained(template_tokenizer_mapping[template])
    processor = AutoProcessor.from_pretrained(template_tokenizer_mapping[template])
    official_prompt = tokenizer.apply_chat_template(messages
    , tokenize=False, add_generation_prompt=add_generation_prompt)
    image_inputs, video_inputs = process_vision_info(messages)
    official_inputs = processor(
        text=[official_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    implemented_inputs = tokenize_conversation(messages, tokenizer, template, max_length=8192, tools=tools, add_generation_prompt=add_generation_prompt, return_tensors="pt", processor=processor)
    
    official_prompt = tokenizer.decode(official_inputs['input_ids'][0])
    implemented_prompt = tokenizer.decode(implemented_inputs['input_ids'][0])
    print(f"Official prompt image tokens: {official_prompt.count('<|image_pad|>')}\nImplemented prompt image tokens: {implemented_prompt.count('<|image_pad|>')}")
    print(f"Official images: {official_inputs['pixel_values'].shape}\nImplemented images: {implemented_inputs['pixel_values'].shape}")

    assert torch.equal(official_inputs["input_ids"], implemented_inputs["input_ids"]), f"""Offical 
    prompt:\n{official_prompt}\nImplemented prompt:\n{implemented_prompt}"""
    
    assert torch.equal(official_inputs["pixel_values"], implemented_inputs["pixel_values"]), f"""Official pixel values: {official_inputs["pixel_values"].shape}\nImplemented pixel values: {implemented_inputs["pixel_values"].shape}"""

    assert torch.equal(official_inputs["image_grid_thw"], implemented_inputs["image_grid_thw"]), f"""Official image grid thw: {official_inputs["image_grid_thw"]}\nImplemented image grid thw: {implemented_inputs["image_grid_thw"]}"""

    assert implemented_inputs["input_ids"].shape == implemented_inputs["action_mask"].shape, f"""Official action mask shape: {official_inputs["action_mask"].shape}\nImplemented action mask shape: {implemented_inputs["action_mask"].shape}"""

    
    print(f"official_prompt: {official_prompt}\nimplemented_prompt: {tokenizer.decode(implemented_inputs['input_ids'][0])}\nofficial_inputs: {official_inputs.keys()}\nimplemented_inputs: {implemented_inputs.keys()}\n")