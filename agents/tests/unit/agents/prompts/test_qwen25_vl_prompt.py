from agents.agents.agents.templates.templates import get_conv_template
from agents.agents.templates.utils import compare_hf_template, format_conversation
from transformers import AutoProcessor, AutoTokenizer

def test_simple_prompts():
    messages = [
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
    ]
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    is_equal, official_prompt, implemented_prompt, highlighted_prompt = compare_hf_template(processor, "qwen2.5-vl", messages=messages, add_generation_prompt=True)
    assert is_equal, f"Official prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    print(f"Highlighted prompt:\n\n{highlighted_prompt}")


def test_simple_prompts_with_system():
    messages = [
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
    ]
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    is_equal, official_prompt, implemented_prompt, highlighted_prompt = compare_hf_template(processor, "qwen2.5-vl", messages=messages, add_generation_prompt=True)
    assert is_equal, f"Official prompt:\n\n{official_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}"
    print(f"Highlighted prompt:\n\n{highlighted_prompt}")


def test_jinja_template():
    conv = get_conv_template("qwen2.5-vl")
    jinja_template = conv.get_jinja_template()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    messages = [
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
            "role": "tool",
            "content": [
                {
                    "type": "text",
                    "text": "Example tool response.",
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
    ]
    official_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokenizer.chat_template = jinja_template
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print(official_prompt)
    print(prompt)

    conv = format_conversation(messages, "qwen2.5-vl", add_generation_prompt=True)
    implemented_prompt = conv.get_prompt()
    print(implemented_prompt)