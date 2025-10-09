""" This file is for testing the vision templates that align seamlessly with HF templates. The templates should align on following aspects:
    - The obtained textual prompt should be the same as the one obtained from HF template with all the following options:
        - add_generation_prompt
        - tools
    - The obtained textual prompt should be the same as the one obtained from Jinja template with all the following options:
        - add_generation_prompt
        - tools
To test vision part, the messages should contain at least one image.
"""


from ....templates.utils import compare_hf_template
from transformers import AutoTokenizer
import pytest
# "qwen2.5-think", "qwen2.5", "qwen2.5-no-tool",
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
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://images.unsplash.com/photo-1592194996308-7b43878e84a6"},
                {"type": "text", "text": "Describe these images."},
                {"type": "image", "image": "https://images.unsplash.com/photo-1599158164704-ef1ec0c94b1c"}
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
        "qwen2.5-vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    }
    tokenizer = AutoTokenizer.from_pretrained(template_tokenizer_mapping[template])

    is_equal, is_equal_between_implemented_prompts, is_equal_between_jinja_prompts, official_prompt, implemented_prompt, implemented_jinja_prompt, highlighted_prompt = compare_hf_template(tokenizer, template, messages=messages, tools=tools,add_generation_prompt=add_generation_prompt)
    assert is_equal, f"Template: {template}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nOfficial prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    assert is_equal_between_jinja_prompts, f"Template: {template}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}\n\nJinja prompt:\n\n{implemented_jinja_prompt}"
    print(f"Official prompt:\n\n{official_prompt}")
    print(f"Highlighted prompt:\n\n{highlighted_prompt}")

