"""Cutting a turn after its tool tag, running an unclosed tag, and dropping old turns when the context is full.

- ``truncate_history_at_action``: the content is cut after the first closing tool tag and the sampled
  ``token_ids`` are cut to match (``trim_token_ids_to_content``), so the next prompt holds the cut text.
- ``close_unclosed_tags``: a tool tag opened but never closed is parsed as if closed at the end.
- ``on_context_overflow="drop_oldest"``: the chain drops its oldest turns instead of ending, keeps each
  pre-drop view as a segment, marks the views that repeat turns, and conversion refuses to train on them.
"""

import json

import pytest
from test_action_agent_tags import make_agent, tool  # noqa: F401
from test_rollout_loop import StrictContext, drain, new_chain_state
from test_rollout_segments import FakeAgent, fake_generation, isolated_runtime  # noqa: F401

from agentfly.agents.rollout import conversion
from agentfly.agents.rollout.events import FinishReason
from agentfly.agents.rollout.loop.generation import CONTENT_IS_PREFIX, trim_token_ids_to_content
from agentfly.agents.rollout.strategies import chain_rollout
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.types import RunResult, Segment, Trajectory
from agentfly.agents.utils.action_format import message_text

PYTHON = tool("python", {"code": {"type": "string"}}, ["code"])


class PieceTokenizer:
    """Token id i decodes to ``pieces[i]``."""

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids)


def text_of(message):
    return message["content"][0]["text"]


# ---- cutting the sampled ids ----


def test_marked_content_cuts_the_ids_at_the_boundary_token():
    pieces = ["<python>", "x=1", "</python>\n", "tail", " more"]
    full = "".join(pieces)
    message = {"content": [{"type": "text", "text": "<python>x=1</python>"}], "token_ids": [0, 1, 2, 3, 4],
               CONTENT_IS_PREFIX: True}
    removed = trim_token_ids_to_content([message], [full], PieceTokenizer(pieces))
    assert removed == 2
    assert message["token_ids"] == [0, 1, 2]
    assert text_of(message) == "<python>x=1</python>\n"  # the boundary token's text, so text and ids agree
    assert CONTENT_IS_PREFIX not in message


def test_unmarked_content_keeps_its_ids():
    """A native tool-call parser also keeps a prefix as content; its ids must stay whole."""
    pieces = ["Let me call it.", '<tool_call>{"name": "f"}</tool_call>']
    message = {"content": [{"type": "text", "text": "Let me call it."}], "token_ids": [0, 1]}
    assert trim_token_ids_to_content([message], ["".join(pieces)], PieceTokenizer(pieces)) == 0
    assert message["token_ids"] == [0, 1]


# ---- ActionAgent parsing ----


def test_truncate_history_at_action_marks_the_cut(make_agent):  # noqa: F811
    agent = make_agent([PYTHON])
    agent.truncate_history_at_action = True
    message = agent._parse_single_response("<python>a = 1</python> and then <python>b</python>", [])
    assert text_of(message) == "<python>a = 1</python>"
    assert message[CONTENT_IS_PREFIX] is True
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"code": "a = 1"}

    whole = agent._parse_single_response("<python>a = 1</python>", [])
    assert CONTENT_IS_PREFIX not in whole  # nothing was cut


def test_without_truncation_the_content_is_the_full_text(make_agent):  # noqa: F811
    agent = make_agent([PYTHON])
    message = agent._parse_single_response("<python>a</python> junk", [])
    assert text_of(message) == "<python>a</python> junk"
    assert CONTENT_IS_PREFIX not in message


@pytest.mark.parametrize("close", [True, False])
def test_an_unclosed_tag_runs_only_when_asked(make_agent, close):  # noqa: F811
    agent = make_agent([PYTHON])
    agent.close_unclosed_tags = close
    text = "<thought>craft it</thought>\n<python>craft({'a': 1}, ('b', 1))"
    message = agent._parse_single_response(text, [])
    assert text_of(message) == text  # stored as sampled
    calls = [json.loads(c["function"]["arguments"]) for c in message["tool_calls"]]
    assert calls == ([{"code": "craft({'a': 1}, ('b', 1))"}] if close else [])


def test_a_closed_tag_is_not_changed_by_close_unclosed_tags(make_agent):  # noqa: F811
    agent = make_agent([PYTHON])
    agent.close_unclosed_tags = True
    message = agent._parse_single_response("<python>x</python><python>y", [])
    assert [json.loads(c["function"]["arguments"]) for c in message["tool_calls"]] == [{"code": "x"}]


# ---- dropping old turns ----


def counting_runtime(monkeypatch):
    """Ten tokens per message; a generation's total is its prompt plus ten."""

    def estimate(agent, step, tools=None, cap=True):
        return 10 * len(step.messages.messages)

    async def generation(agent, chain, current_step, *args, **kwargs):
        message, _ = await fake_generation(agent, chain, current_step, *args, **kwargs)
        return message, estimate(agent, current_step) + 10

    monkeypatch.setattr(chain_rollout, "estimate_chat_prompt_tokens", estimate)
    monkeypatch.setattr(chain_rollout, "generate_response", generation)


def small_agent(turns):
    agent = FakeAgent([["act"]] * turns)
    agent.max_model_len = chain_rollout.TOKEN_SAFETY_MARGIN + 60  # room for about two turns
    return agent


@pytest.mark.asyncio
async def test_by_default_a_full_context_ends_the_episode(isolated_runtime, monkeypatch):  # noqa: F811
    counting_runtime(monkeypatch)
    agent, state = small_agent(5), new_chain_state()
    await drain(ChainRollout().run_loop(
        agent=agent, context=StrictContext(), state=state, tools=[], max_turns=5, generation_config={},
    ))
    assert state.chain.info["finish_reason"] == FinishReason.MAX_MODEL_LEN
    assert agent.tool_calls < 5


@pytest.mark.asyncio
async def test_drop_oldest_keeps_the_task_and_goes_on(isolated_runtime, monkeypatch):  # noqa: F811
    counting_runtime(monkeypatch)
    agent, state = small_agent(5), new_chain_state()
    strategy = ChainRollout(on_context_overflow="drop_oldest")
    await drain(strategy.run_loop(
        agent=agent, context=StrictContext(), state=state, tools=[], max_turns=5, generation_config={},
    ))
    assert state.chain.info["finish_reason"] == FinishReason.MAX_TURNS
    assert agent.tool_calls == 5
    drops = state.chain.info["context_drops"]
    assert drops >= 1

    segments = strategy.segments(state)
    assert len(segments) == drops + 1
    assert [s.repeats_turns for s in segments] == [False] + [True] * drops
    for segment in segments:  # every view starts with the task
        assert [message_text(m) for m in segment.messages[:2]] == ["SYS", "O0"]
    final = [message_text(m) for m in segments[-1].messages]
    assert "O1" not in final and "O5" in final  # the oldest turn is gone, the newest is there


def test_conversion_refuses_views_that_repeat_turns():
    messages = [{"role": "user", "content": "O0"}, {"role": "assistant", "content": "A1"}]
    result = RunResult(
        trajectories=[Trajectory(segments=[Segment(messages=messages), Segment(messages=messages, repeats_turns=True)])],
        rollout="chain",
    )
    with pytest.raises(ValueError, match="trained twice"):
        conversion.build_segment_batch(None, result, tokenize=lambda *a, **k: pytest.fail("must not tokenize"))


def test_unknown_overflow_mode_is_rejected():
    with pytest.raises(ValueError, match="on_context_overflow"):
        ChainRollout(on_context_overflow="summarize")
