"""``reasoning_parser_name``: a generated turn is stored as ``reasoning_content`` + ``content``.

K2-Horizon's chat template requires a separate reasoning field on every assistant turn (it never reads the
reasoning out of ``content``), so a turn stored as one text does not render. The invariant pinned here is the one
training relies on: the history rendered from the stored messages is the token sequence the model was sampled on.
Needs the K2-Horizon tokenizer in the local HF cache and a vLLM with the ``k2_horizon`` parsers (>= 0.30).
"""
import json
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("chat_bricks")

from agentfly.agents.agent_base import BaseAgent  # noqa: E402
from agentfly.agents.utils.tokenizer import tokenize_trajectories  # noqa: E402
from agentfly.agents.rollout.loop.generation import (  # noqa: E402
    CONTENT_IS_PREFIX,
    REASONING_UNCLOSED,
    generate_response,
    split_reasoning_responses,
    trim_token_ids_to_content,
)

K2 = "IFM/K2-Horizon-MoVA-36B-A4B"
IM_END = "<|ifm|im_end|>"
CALL = (
    "<ifm|tool_calls>\n<ifm|tool_call>bash\n<ifm|arg_key>command</ifm|arg_key>\n<ifm|arg_value>{}</ifm|arg_value>\n"
    "</ifm|tool_call>\n</ifm|tool_calls>"
)
# What the model writes after a generation prompt that ends with "<ifm|think>\n".
RAW = "List the files first.</ifm|think>Let me look." + CALL.format("ls -la")


@pytest.fixture(scope="module")
def tok():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
        from vllm.reasoning import ReasoningParserManager

        ReasoningParserManager.get_reasoning_parser("k2_horizon")
        tokenizer = AutoTokenizer.from_pretrained(K2, trust_remote_code=True)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"K2-Horizon tokenizer or vLLM k2_horizon parser unavailable: {e}")
    tokenizer.pad_token = tokenizer.eos_token  # K2 defines no pad token; batching needs one
    return tokenizer


@pytest.fixture
def make_agent(monkeypatch, tok):
    import agentfly.agents.agent_base as agent_base
    from agentfly.agents.specialized.miniswe.agent import MinisweAgent

    monkeypatch.setattr(agent_base, "create_tokenizer", lambda model: tok)
    monkeypatch.setattr(agent_base, "create_processor", lambda model: None)
    monkeypatch.setattr(agent_base, "get_jinja_template", lambda template: None)
    monkeypatch.setattr(BaseAgent, "_init_llm_engine", lambda *args: None)

    def _make(**kwargs):
        kwargs.setdefault("tool_parser_name", "k2_horizon")
        kwargs.setdefault("reasoning_parser_name", "k2_horizon")
        kwargs.setdefault("backend_config", {"backend": "async_vllm"})
        return MinisweAgent(K2, monitors=[], **kwargs)

    return _make


def _ids(tok, text, terminated=True):
    return tok.encode(text + (IM_END if terminated else ""), add_special_tokens=False)


async def _turn(agent, history, response):
    """One generation step on ``history`` with the backend's reply faked."""

    async def generate_async(messages, **kwargs):
        return response

    agent.generate_async = generate_async
    step = SimpleNamespace(messages=SimpleNamespace(messages=history))
    msg, _ = await generate_response(agent, None, step, agent.tools, 0, "c0", {}, SimpleNamespace(metadata={}))
    return msg


def _prompt_ids(agent, tok, messages):
    """The prompt for the next turn, as training tokenizes it (AgentFly's own path)."""
    out = tokenize_trajectories(agent, [messages], tokenizer=tok)
    return out["input_ids"][0].tolist()


# ---- start-up check ----


def test_template_that_needs_the_field_requires_a_parser(make_agent):
    with pytest.raises(ValueError, match="reasoning_parser_name"):
        make_agent(reasoning_parser_name=None)


def test_a_server_that_returns_the_field_needs_no_parser(make_agent):
    make_agent(reasoning_parser_name=None, backend_config={"backend": "client"})


def test_template_that_ignores_the_field_is_refused(make_agent):
    import agentfly.templates  # noqa: F401  (registers qwen3-miniswe, which renders content only)

    with pytest.raises(ValueError, match="does not render reasoning_content"):
        make_agent(template="qwen3-miniswe")


def test_parser_and_strip_reasoning_exclude_each_other(make_agent):
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.check_settings({**agent.own_settings, "strip_reasoning": True})


def test_subagents_inherit_the_parser(make_agent):
    agent = make_agent(subagents={"helper": {"system_prompt": "help"}})
    assert agent.create_subagent("helper").reasoning_parser is not None


# ---- the stored turn ----


@pytest.mark.asyncio
async def test_turn_is_stored_as_two_fields(make_agent, tok):
    sampled = _ids(tok, RAW)
    msg = await _turn(make_agent(), [{"role": "user", "content": "q"}], {"response_texts": [RAW], "response_ids": [sampled]})
    assert msg["reasoning_content"] == "List the files first."
    assert msg["content"][0]["text"] == "Let me look." + CALL.format("ls -la")
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"command": "ls -la"}
    assert msg["token_ids"] == sampled and REASONING_UNCLOSED not in msg


@pytest.mark.asyncio
async def test_calls_inside_the_reasoning_are_not_run(make_agent):
    raw = "Maybe " + CALL.format("rm -rf /") + " no.</ifm|think>" + CALL.format("ls")
    msg = await _turn(make_agent(), [{"role": "user", "content": "q"}], {"response_texts": [raw]})
    assert [json.loads(c["function"]["arguments"]) for c in msg["tool_calls"]] == [{"command": "ls"}]


@pytest.mark.asyncio
async def test_reasoning_returned_by_the_backend_is_kept(make_agent):
    answer = "Let me look." + CALL.format("ls")
    response = {
        "response_texts": [answer],
        "response_dict": {"choices": [{"message": {"content": answer, "reasoning_content": "from the server"}}]},
    }
    msg = await _turn(make_agent(), [{"role": "user", "content": "q"}], response)
    assert msg["reasoning_content"] == "from the server" and msg["content"][0]["text"] == answer


# ---- the history is the sequence the model was sampled on ----


@pytest.mark.asyncio
@pytest.mark.parametrize("with_ids", [True, False])
async def test_history_continues_the_sampled_sequence(make_agent, tok, with_ids):
    """Two turns: prompt(history + turn + observation) == prompt(history) + sampled ids + the observation."""
    agent = make_agent()
    history = [{"role": "user", "content": "q"}]
    observation = "<|ifm|im_start|>tool\nobs" + IM_END + "<|ifm|im_start|>assistant\n<ifm|think>\n"
    for raw in (RAW, "Now read it.</ifm|think>" + CALL.format("cat a.py")):
        sampled = _ids(tok, raw)
        response = {"response_texts": [raw], "response_ids": [sampled]} if with_ids else {"response_texts": [raw]}
        before = _prompt_ids(agent, tok, history)
        msg = await _turn(agent, history, response)
        history = history + [msg, {"role": "tool", "content": "obs"}]
        expected = before + sampled + tok.encode(observation, add_special_tokens=False)
        assert _prompt_ids(agent, tok, history) == expected


@pytest.mark.asyncio
async def test_turn_cut_inside_its_reasoning_keeps_the_sampled_ids(make_agent, tok):
    """No closing tag and no terminator (token limit): with the ids, the history is still what was sampled."""
    agent = make_agent()
    raw = "I was still thinking about"
    sampled = _ids(tok, raw, terminated=False)
    history = [{"role": "user", "content": "q"}]
    before = _prompt_ids(agent, tok, history)
    msg = await _turn(agent, history, {"response_texts": [raw], "response_ids": [sampled]})
    assert msg[REASONING_UNCLOSED] is True
    after = _prompt_ids(agent, tok, history + [msg, {"role": "tool", "content": "obs"}])
    tail = IM_END + "<|ifm|im_start|>tool\nobs" + IM_END + "<|ifm|im_start|>assistant\n<ifm|think>\n"
    assert after == before + sampled + tok.encode(tail, add_special_tokens=False)


# ---- cutting the ids of a turn kept only up to its first action ----


def test_ids_are_cut_to_reasoning_plus_kept_answer(make_agent, tok):
    raw = "Think.</ifm|think>Go.<x>1</x> and more text"
    texts, splits = split_reasoning_responses(make_agent(), [raw], {})
    assert texts == ["Go.<x>1</x> and more text"] and splits[0]["head"] == "Think.</ifm|think>"
    msg = {"role": "assistant", "content": "Go.<x>1</x>", CONTENT_IS_PREFIX: True, "token_ids": _ids(tok, raw)}
    removed = trim_token_ids_to_content([msg], [raw], tok, splits)
    assert removed > 0 and msg["content"] == "Go.<x>1</x>"
    assert tok.decode(msg["token_ids"]) == "Think.</ifm|think>Go.<x>1</x>"


def test_without_a_parser_responses_are_left_whole():
    agent = SimpleNamespace(reasoning_parser=None)
    assert split_reasoning_responses(agent, ["a</think>b"], {}) == (["a</think>b"], None)
