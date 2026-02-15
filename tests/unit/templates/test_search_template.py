from agentfly.templates import *
from chat_bricks import Chat, get_template, Template, tokenize_conversations
import pytest
from transformers import AutoTokenizer

@pytest.mark.parametrize("template_name", ["search-r1"])
@pytest.mark.parametrize("messages", [
    [
        {"role": "user", "content": "Who is the president of the United States?"},
        {"role": "assistant", "content": "<think>I need to search for the president of the United States.</think>\n<search>president of the United States</search>\n"},
        {"role": "tool", "content": "(US President): President is a political office in the United States... The current president is Joe Biden."},
        {"role": "assistant", "content": "I found the president of the United States is Joe Biden. <answer>Joe Biden</answer>"},
    ],
])
@pytest.mark.parametrize("tools", [
    None,
])
def test_search_template(template_name, messages, tools):

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)

    chat = Chat(template_name, messages, tools=tools)
    prompt = chat.prompt()
    print(f"Prompt: {prompt}")
    print(f"Prompt with mask: {chat.prompt_with_mask()}")
    
    inputs = tokenize_conversations(
        [messages],
        tokenizer,
        template_name,
        max_length=2048,
        tools=tools,
    )
    print(f"Inputs: {inputs}")

    print(f"Decoded prompt: {tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}")