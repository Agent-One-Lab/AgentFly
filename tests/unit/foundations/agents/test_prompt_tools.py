"""``render_tools_in_prompt`` / ``prompt_tools()``: one source of truth for the tool
schemas rendered into the prompt, consulted by BOTH generation and training
tokenization — so the model is trained under the prompt it was sampled with.

Background: the ActionAgent puts the environment interface in the prompt text
(``<action>`` format) and keeps its tool only to execute the parsed action. The
backend nevertheless rendered the tool schema into the sampling prompt (Qwen's
``# Tools ... <tools>`` block, ~164 tokens) while ``tokenize_trajectories`` rendered
none — a sampling/training prompt mismatch. These tests pin the fix.
"""
import types

import pytest

from agentfly.agents.agent_base import BaseAgent
from agentfly.agents.utils.tokenizer import tokenize_trajectories

HF_TMPL = "Qwen/Qwen2.5-1.5B-Instruct"

_SCHEMA = {
    "type": "function",
    "function": {
        "name": "alfworld_step",
        "description": "Take an action.",
        "parameters": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]},
    },
}


def _fake_tool():
    return types.SimpleNamespace(name="alfworld_step", schema=_SCHEMA)


def _agent_ns(render: bool, tools=None):
    """A BaseAgent-shaped namespace so we can call the unbound BaseAgent.prompt_tools."""
    return types.SimpleNamespace(render_tools_in_prompt=render, tools=[_fake_tool()] if tools is None else tools)


# ---- BaseAgent.prompt_tools -------------------------------------------------

def test_prompt_tools_returns_schemas_when_rendering():
    assert BaseAgent.prompt_tools(_agent_ns(True)) == [_SCHEMA]


def test_prompt_tools_empty_when_not_rendering():
    assert BaseAgent.prompt_tools(_agent_ns(False)) == []


def test_prompt_tools_empty_when_no_tools():
    # Always a list (never None) so it flows through numpy/DataProto columns.
    assert BaseAgent.prompt_tools(_agent_ns(True, tools=[])) == []


# ---- ActionAgent defaults to NOT rendering tools -----------------------------

def test_action_agent_defaults_render_tools_off(monkeypatch):
    from agentfly.agents.specialized import action_agent as aa

    captured = {}

    def fake_init(self, model_name_or_path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(BaseAgent, "__init__", fake_init)
    aa.ActionAgent("dummy-model", tools=[_fake_tool()])
    assert captured["render_tools_in_prompt"] is False

    captured.clear()
    aa.ActionAgent("dummy-model", tools=[_fake_tool()], render_tools_in_prompt=True)  # explicit override wins
    assert captured["render_tools_in_prompt"] is True


# ---- training tokenization consults prompt_tools -----------------------------

def _real_tokenizer():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    return transformers.AutoTokenizer.from_pretrained(HF_TMPL)


def _agent_for_tokenize(tok, prompt_tools):
    return types.SimpleNamespace(
        tokenizer=tok, processor=None, template=HF_TMPL, max_model_len=4096, prompt_tools=prompt_tools
    )


MSGS = [[{"role": "user", "content": "obs"}, {"role": "assistant", "content": "<think>a</think><action>look</action>"}]]


def test_tokenize_trajectories_passes_agent_prompt_tools(monkeypatch):
    import agentfly.agents.utils.tokenizer as tokenizer_utils

    seen = {}

    def fake_tokenize_conversations(messages_list, **kw):
        import torch
        seen.update(kw)
        n, L = len(messages_list), 4
        return {"input_ids": torch.ones(n, L, dtype=torch.long), "attention_mask": torch.ones(n, L, dtype=torch.long),
                "labels": torch.ones(n, L, dtype=torch.long), "action_mask": torch.ones(n, L, dtype=torch.long)}

    monkeypatch.setattr(tokenizer_utils, "tokenize_conversations", fake_tokenize_conversations)
    tok = _real_tokenizer()
    tokenize_trajectories(_agent_for_tokenize(tok, lambda: [_SCHEMA]), messages_list=MSGS, tokenizer=tok)
    assert seen["tools"] == [_SCHEMA]
    tokenize_trajectories(_agent_for_tokenize(tok, lambda: []), messages_list=MSGS, tokenizer=tok)
    assert seen["tools"] is None  # empty -> no tools block


def test_training_render_has_no_tools_block_for_action_agent():
    tok = _real_tokenizer()
    out = tokenize_trajectories(_agent_for_tokenize(tok, lambda: []), messages_list=MSGS, tokenizer=tok)
    text = tok.decode(out["input_ids"][0], skip_special_tokens=False)
    assert "<tools>" not in text and "# Tools" not in text


def test_training_render_has_tools_block_for_tool_calling_agent():
    tok = _real_tokenizer()
    out = tokenize_trajectories(
        _agent_for_tokenize(tok, lambda: [_SCHEMA]), messages_list=MSGS, tokenizer=tok
    )
    text = tok.decode(out["input_ids"][0], skip_special_tokens=False)
    assert "<tools>" in text and "alfworld_step" in text
    # and it is exactly what the tokenizer's own template renders at sampling time
    sampled = tok.apply_chat_template(MSGS[0][:1], tools=[_SCHEMA], tokenize=False, add_generation_prompt=True)
    assert text.startswith(sampled)


# ---- backend guard: sampled prompt ids vs training render --------------------

def _backend(tok):
    from agentfly.utils.llm_backends.llm_backends import AsyncVerlBackend

    b = object.__new__(AsyncVerlBackend)
    b.model_name, b.template, b.tokenizer = HF_TMPL, None, tok
    b.check_prompt_consistency = True
    b._prompt_check_calls, b.prompt_check_stats = 0, {"checked": 0, "mismatched": 0}
    return b


def _gb_from_prompt_ids(rows, pad_to=None):
    import torch
    pad_to = pad_to or max(len(r) for r in rows)
    prompts = torch.zeros(len(rows), pad_to, dtype=torch.long)
    am = torch.zeros(len(rows), pad_to + 2, dtype=torch.long)  # + a 2-token "response"
    for i, r in enumerate(rows):
        prompts[i, pad_to - len(r):] = torch.tensor(r)   # left-padded like verl
        am[i, pad_to - len(r):] = 1
    return {"prompts": prompts, "attention_mask": am}


def test_prompt_guard_passes_when_identical():
    from chat_bricks import Chat
    tok = _real_tokenizer()
    msgs = [[{"role": "user", "content": "hi"}]]
    ids = Chat(template=HF_TMPL, messages=msgs[0], tokenizer=tok).tokenize(tok, add_generation_prompt=True)["input_ids"][0].tolist()
    b = _backend(tok)
    b._check_prompt_consistency(None, msgs, [], _gb_from_prompt_ids([ids]))
    assert b.prompt_check_stats == {"checked": 1, "mismatched": 0}


def test_prompt_guard_flags_tools_block_mismatch():
    tok = _real_tokenizer()
    msgs = [[{"role": "user", "content": "hi"}]]
    # what the OLD path sampled with: the tools block rendered in
    sampled = tok.apply_chat_template(msgs[0], tools=[_SCHEMA], tokenize=True, add_generation_prompt=True, return_dict=False)
    b = _backend(tok)
    b._check_prompt_consistency(None, msgs, [], _gb_from_prompt_ids([list(sampled)]))  # training renders NO tools
    assert b.prompt_check_stats == {"checked": 1, "mismatched": 1}


def test_react_agent_defaults_render_tools_off(monkeypatch):
    # ReactAgent puts the schemas in its system prompt TEXT, so it must not also
    # render the chat template's tools block.
    from agentfly.agents.specialized import react_agent as ra

    captured = {}

    def fake_init(self, model_name_or_path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(BaseAgent, "__init__", fake_init)
    ra.ReactAgent("dummy-model", tools=[_fake_tool()])
    assert captured["render_tools_in_prompt"] is False
    assert "alfworld_step" in captured["system_prompt"]  # schemas live in the prompt text


def test_prompt_guard_can_be_switched_off():
    # backend_config.check_prompt_consistency=false -> no check runs, even on a mismatch
    tok = _real_tokenizer()
    msgs = [[{"role": "user", "content": "hi"}]]
    sampled = tok.apply_chat_template(msgs[0], tools=[_SCHEMA], tokenize=True, add_generation_prompt=True, return_dict=False)
    b = _backend(tok)
    b.check_prompt_consistency = False
    b._check_prompt_consistency(None, msgs, [], _gb_from_prompt_ids([list(sampled)]))
    assert b.prompt_check_stats == {"checked": 0, "mismatched": 0}
    assert b._prompt_check_calls == 0
