import pytest
from transformers import AutoTokenizer
from agentfly.templates.templates import Chat, get_template


@pytest.mark.parametrize("template_name", ["qwen2.5-vl-system-tool"])
@pytest.mark.parametrize(
    "messages",
    [
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
            },
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://images.unsplash.com/photo-1592194996308-7b43878e84a6",
                    },
                    {"type": "text", "text": "Describe these images."},
                    {
                        "type": "image",
                        "image": "https://images.unsplash.com/photo-1599158164704-ef1ec0c94b1c",
                    },
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
                "role": "tool",
                "content": [
                    {"type": "text", "text": "Image Id: 000000"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://images.unsplash.com/photo-1592194996308-7b43878e84a6"
                        },
                    },
                ],
            },
        ],
    ],
)
@pytest.mark.parametrize(
    "tools",
    [
        [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "A function that multiplies two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "The first number to multiply",
                            },
                            "y": {
                                "type": "number",
                                "description": "The second number to multiply",
                            },
                        },
                        "required": ["x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "A function that multiplies two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "The first number to multiply",
                            },
                            "y": {
                                "type": "number",
                                "description": "The second number to multiply",
                            },
                        },
                        "required": ["x", "y"],
                    },
                },
            },
        ]
    ],
)
def test_template_openai_format(template_name, messages, tools):
    template_tokenizer_mapping = {
        "qwen2.5-vl": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen2.5-vl-system-tool": "Qwen/Qwen2.5-VL-3B-Instruct",
    }
    tokenizer = AutoTokenizer.from_pretrained(template_tokenizer_mapping[template_name])
    template = get_template(template_name)
    tokenizer.chat_template = template.jinja_template()

    jinja_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tools=tools, tokenize=False
    )
    print(jinja_prompt)

    chat = Chat(template_name, messages, tools=tools)
    implemented_prompt = chat.prompt(add_generation_prompt=False, tools=tools)
    print(implemented_prompt)

    assert jinja_prompt == implemented_prompt
