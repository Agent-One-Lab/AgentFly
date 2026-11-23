"""action-agent-qwen35: Qwen3.5 in non-thinking mode, with the same context at generation and in training.

Every assistant turn of a training row must sit after exactly the tokens its generation prompt had, and the
trained tokens must be the assistant replies only (the empty think block is part of the prompt).
"""

import pytest

pytest.importorskip("chat_bricks")
from chat_bricks import Chat  # noqa: E402

import agentfly.templates  # noqa: E402,F401  (registers the template)

TEMPLATE = "action-agent-qwen35"
EMPTY_THINK = "<think>\n\n</think>\n\n"
REPLIES = ["<thought>check</thought>\n<python>get_info(['a1_i5'])</python>", "<thought>done</thought>\n<python>finish('ok')</python>"]
MESSAGES = [
    {"role": "system", "content": "You are a crafting agent."},
    {"role": "user", "content": "Craft 2x a1_i5."},
    {"role": "assistant", "content": REPLIES[0]},
    {"role": "user", "content": "<observation>[...]</observation>"},
    {"role": "assistant", "content": REPLIES[1]},
]


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen/Qwen3.5-4B tokenizer not in the local HF cache")


def ids_of(inputs, key):
    value = inputs[key]
    return value[0].tolist() if hasattr(value, "tolist") else list(value)


def test_generation_prompt_ends_with_empty_think_block():
    prompt = Chat(TEMPLATE, MESSAGES[:2]).prompt(add_generation_prompt=True)
    assert prompt.endswith("<|im_start|>assistant\n" + EMPTY_THINK)


def test_history_turns_keep_the_empty_think_block():
    prompt = Chat(TEMPLATE, MESSAGES[:4]).prompt(add_generation_prompt=True)
    assert prompt.count(EMPTY_THINK) == 2  # the earlier reply and the new generation prompt
    assert "<|im_start|>assistant\n" + EMPTY_THINK + REPLIES[0] + "<|im_end|>" in prompt


def test_trained_tokens_are_the_replies_only(tokenizer):
    inputs = Chat(TEMPLATE, MESSAGES, tokenizer=tokenizer).tokenize(tokenizer, add_generation_prompt=False)
    ids, mask = ids_of(inputs, "input_ids"), ids_of(inputs, "action_mask")
    trained = tokenizer.decode([i for i, m in zip(ids, mask) if m])
    assert "<think>" not in trained and "</think>" not in trained
    for reply in REPLIES:
        assert reply in trained
    assert trained.replace("<|im_end|>", "").replace("\n", "") == "".join(REPLIES).replace("\n", "")


def test_each_turn_is_trained_in_its_generation_context(tokenizer):
    inputs = Chat(TEMPLATE, MESSAGES, tokenizer=tokenizer).tokenize(tokenizer, add_generation_prompt=False)
    ids, mask = ids_of(inputs, "input_ids"), ids_of(inputs, "action_mask")
    starts = [i for i in range(len(mask)) if mask[i] and (i == 0 or not mask[i - 1])]
    assert len(starts) == 2
    for start, n_before in zip(starts, (2, 4)):
        prompt = Chat(TEMPLATE, MESSAGES[:n_before]).prompt(add_generation_prompt=True)
        assert ids[:start] == tokenizer(prompt, add_special_tokens=False)["input_ids"]
