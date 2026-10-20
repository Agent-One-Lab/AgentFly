"""Format check when the chat template opens ``<think>`` in the generation prompt.

Native-thinking templates (e.g. Qwen3.5-4B) end the generation prompt with ``<think>\\n``,
so a correct response carries only ``...</think>`` followed by the action. Counting the
prompt's opening tag keeps those turns format-valid; templates that do not open a think
block are checked exactly as before.
"""

from types import SimpleNamespace

import pytest

from agentfly.agents.utils.action_format import action_format_valid, generation_opens_think

NATIVE = "The user wants a red shirt.\n</think>\n\n<action>search[red shirt]</action>"
TAGGED = "<think>The user wants a red shirt.</think>\n<action>search[red shirt]</action>"


def test_native_think_response_is_valid_only_when_the_prompt_opened_think():
    assert action_format_valid(NATIVE) == 0.0
    assert action_format_valid(NATIVE, think_opened=True) == 1.0


def test_think_opened_keeps_other_format_rules():
    assert action_format_valid(TAGGED, think_opened=True) == 1.0
    assert action_format_valid("reasoning\n</think>\n\nsearch[x]", think_opened=True) == 0.0  # no <action>
    assert action_format_valid("reasoning <action>search[x]</action>", think_opened=True) == 0.0  # no </think>
    assert action_format_valid("推理\n</think>\n<action>search[x]</action>", think_opened=True) == 0.0  # Chinese
    assert action_format_valid("", think_opened=True) == 0.0


@pytest.mark.parametrize(
    "model, expected",
    [
        ("Qwen/Qwen3.5-4B", True),        # native thinking: generation prompt ends with <think>
        ("Qwen/Qwen3.5-2B", False),       # thinking off by default: empty <think></think> block
        ("Qwen/Qwen2.5-1.5B-Instruct", False),
    ],
)
def test_generation_opens_think(model, expected):
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{model} tokenizer unavailable: {e}")
    agent = SimpleNamespace(template=model, tokenizer=tokenizer)
    assert generation_opens_think(agent) is expected
    assert agent._generation_opens_think is expected  # cached for later conversions
