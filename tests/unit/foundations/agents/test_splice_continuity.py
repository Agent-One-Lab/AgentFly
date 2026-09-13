"""Backend guard: a sampled turn must reappear in later prompts as exactly
(turn prompt + sampled ids). Exercises ``AsyncVerlBackend._check_splice_continuity``
against the real Qwen3.5 thinking template, where the old splice inserted an
empty-think closer between the generation prompt and the sampled ids.
"""
import os

import pytest

pytest.importorskip("chat_bricks")
transformers = pytest.importorskip("transformers")

from agentfly.utils.llm_backends.llm_backends import AsyncVerlBackend

QWEN35 = "Qwen/Qwen3.5-9B"
BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}


@pytest.fixture(scope="module")
def tok():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return transformers.AutoTokenizer.from_pretrained(QWEN35, trust_remote_code=True)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{QWEN35} tokenizer unavailable offline: {e}")


def _backend(tok):
    b = object.__new__(AsyncVerlBackend)
    b.model_name, b.template, b.tokenizer = QWEN35, QWEN35, tok
    b.check_prompt_consistency = True
    b._prompt_check_calls = 1  # "first call" -> the guard runs
    b.prompt_check_stats = {"checked": 0, "mismatched": 0, "continuity_checked": 0, "continuity_broken": 0}
    b._warned_vision_fallback = False
    return b


def _sample(tok, think, content, cmd):
    text = (
        f"{think}\n</think>\n\n{content}\n<tool_call>\n<function=bash>\n<parameter=command>\n"
        f"{cmd}\n</parameter>\n</function>\n</tool_call>"
    )
    return text, tok.encode(text, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]


def _messages(tok, ids1=None):
    text1, sampled = _sample(tok, "THOUGHT_ONE", "I will list files.", "ls")
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "do the task"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": text1}],
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}],
            "loss": True,
            "status": "continue",
            "token_ids": ids1 if ids1 is not None else sampled,
        },
        {"role": "tool", "tool_call_id": "c1", "tool_name": "bash", "content": [{"type": "text", "text": "file_a"}]},
    ]


def test_spliced_turn_is_continuous_with_its_prompt(tok):
    b = _backend(tok)
    msgs = _messages(tok)
    prompt_ids = b._render_prompt_ids([msgs], [BASH])
    b._check_splice_continuity(prompt_ids, [msgs], [BASH])
    assert b.prompt_check_stats["continuity_checked"] == 1
    assert b.prompt_check_stats["continuity_broken"] == 0
    # And explicitly: the prompt is (turn-1 prompt) + (sampled ids) + ...
    turn_prompt = b._render_prompt_ids([msgs[:2]], [BASH])[0]
    ids = msgs[2]["token_ids"]
    assert prompt_ids[0][: len(turn_prompt) + len(ids)] == turn_prompt + ids


def test_guard_flags_a_discontinuity(tok):
    b = _backend(tok)
    msgs = _messages(tok)
    # Corrupt what the backend "sent": insert the old empty-think closer glue
    # right after the generation prompt, before the sampled ids.
    turn_prompt = b._render_prompt_ids([msgs[:2]], [BASH])[0]
    glue = tok.encode("\n</think>\n\n", add_special_tokens=False)
    good = b._render_prompt_ids([msgs], [BASH])[0]
    bad = turn_prompt + glue + good[len(turn_prompt):]
    b._check_splice_continuity([bad], [msgs], [BASH])
    assert b.prompt_check_stats["continuity_checked"] == 1
    assert b.prompt_check_stats["continuity_broken"] == 1


def test_strict_mode_raises(tok):
    b = _backend(tok)
    b.check_prompt_consistency = "strict"
    msgs = _messages(tok)
    good = b._render_prompt_ids([msgs], [BASH])[0]
    bad = list(good)
    bad[len(bad) // 2] = (bad[len(bad) // 2] + 1) % 1000  # corrupt inside the sampled span or prompt
    with pytest.raises(RuntimeError):
        b._check_splice_continuity([bad], [msgs], [BASH])


def test_rows_without_token_ids_are_skipped(tok):
    b = _backend(tok)
    msgs = _messages(tok)
    msgs[2].pop("token_ids")
    b._check_splice_continuity(b._render_prompt_ids([msgs], [BASH]), [msgs], [BASH])
    assert b.prompt_check_stats["continuity_checked"] == 0
