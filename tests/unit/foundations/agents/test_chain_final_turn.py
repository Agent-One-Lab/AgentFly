"""``on_context_overflow="final_turn"`` and ``final_turn_on_max_turns``: one last turn before a run ends.

The agent works within ``max_model_len - TOKEN_SAFETY_MARGIN - final_turn_reserve``; reaching it adds the
final-turn message and allows one more reply (its tool calls run, so a finish tool still ends the run normally),
then the run ends. A tool result that would cut into the reserve is replaced by a note. With
``final_turn_on_max_turns`` the same last turn follows a run's ``max_turns``.
"""

import pytest
from test_rollout_loop import StrictContext, drain, new_chain_state
from test_rollout_segments import FakeAgent, isolated_runtime  # noqa: F401
from test_truncate_and_context_drops import counting_runtime

from agentfly.agents.rollout.events import FinishReason
from agentfly.agents.rollout.strategies import chain_rollout
from agentfly.agents.rollout.strategies.chain_rollout import FINAL_TURN_OMITTED, ChainRollout
from agentfly.agents.utils.action_format import message_text

FINAL = "FINAL: answer now."
RESERVE = 100


def agent_with_room(turns, room=60, terminal_after=None):
    agent = FakeAgent([["act"]] * turns, terminal_after=terminal_after)
    agent.max_model_len = chain_rollout.TOKEN_SAFETY_MARGIN + RESERVE + room
    return agent


def final_turn_strategy(**kwargs):
    return ChainRollout(on_context_overflow="final_turn", final_turn_message=FINAL, final_turn_reserve=RESERVE, **kwargs)


def texts(state):
    return [message_text(m) for m in state.current_step.messages]


async def run(strategy, agent, max_turns):
    state = new_chain_state()
    await drain(strategy.run_loop(
        agent=agent, context=StrictContext(), state=state, tools=[], max_turns=max_turns, generation_config={},
    ))
    return state


@pytest.mark.asyncio
async def test_full_context_gets_one_final_turn_then_ends(isolated_runtime, monkeypatch):  # noqa: F811
    counting_runtime(monkeypatch)
    agent = agent_with_room(30)
    state = await run(final_turn_strategy(), agent, max_turns=30)
    assert state.chain.info["finish_reason"] == FinishReason.MAX_MODEL_LEN
    assert state.chain.info["final_turns"] == 1
    view = texts(state)
    assert view.count(FINAL) == 1
    after = view[view.index(FINAL) + 1 :]
    assert after and after[0].startswith("A")  # the final reply, after the message
    assert "final_turn" not in state.chain.info  # reset for a later run on the same state
    assert agent.tool_calls < 30


@pytest.mark.asyncio
async def test_a_finish_in_the_final_turn_ends_the_run_normally(isolated_runtime, monkeypatch):  # noqa: F811
    counting_runtime(monkeypatch)
    probe = agent_with_room(30)
    await run(final_turn_strategy(), probe, max_turns=30)
    agent = agent_with_room(30, terminal_after=probe.tool_calls)  # the final turn's tool call ends the run
    state = await run(final_turn_strategy(), agent, max_turns=30)
    assert state.chain.info["final_turns"] == 1
    assert state.chain.info["finish_reason"] not in (FinishReason.MAX_MODEL_LEN, FinishReason.MAX_TURNS)
    assert agent.tool_calls == probe.tool_calls


@pytest.mark.asyncio
async def test_end_mode_is_unchanged(isolated_runtime, monkeypatch):  # noqa: F811
    counting_runtime(monkeypatch)
    agent = agent_with_room(30)
    state = await run(ChainRollout(), agent, max_turns=30)
    assert state.chain.info["finish_reason"] == FinishReason.MAX_MODEL_LEN
    assert "final_turns" not in state.chain.info and FINAL not in texts(state)


@pytest.mark.asyncio
async def test_max_turns_final_turn_only_when_asked(isolated_runtime, monkeypatch):  # noqa: F811
    counting_runtime(monkeypatch)
    plain = FakeAgent([["act"]] * 10)
    await run(ChainRollout(), plain, max_turns=3)
    assert plain.tool_calls == 3

    agent = FakeAgent([["act"]] * 10)
    strategy = ChainRollout(final_turn_message=FINAL, final_turn_on_max_turns=True)
    state = await run(strategy, agent, max_turns=3)
    assert agent.tool_calls == 4  # three turns, then the final one
    assert state.chain.info["finish_reason"] == FinishReason.MAX_TURNS
    assert texts(state).count(FINAL) == 1 and state.chain.info["final_turns"] == 1


@pytest.mark.asyncio
async def test_a_result_that_cuts_into_the_reserve_is_omitted(isolated_runtime, monkeypatch):  # noqa: F811
    counting_runtime(monkeypatch)
    # every tool result costs 40 tokens: the second or third one crosses into the reserve
    monkeypatch.setattr(chain_rollout, "observation_token_length", lambda agent, obs: 40 if obs.startswith("O") else 2)
    agent = agent_with_room(30, room=80)
    state = await run(final_turn_strategy(), agent, max_turns=30)
    assert state.chain.info["omitted_results"] == 1  # the next turn is the final one: no repeated omissions
    view = texts(state)
    assert FINAL_TURN_OMITTED in view
    assert view.index(FINAL) == view.index(FINAL_TURN_OMITTED) + 1  # the final message right after the note
    assert state.chain.info["final_turns"] == 1


def test_final_turn_needs_a_message():
    with pytest.raises(ValueError, match="final_turn_message"):
        ChainRollout(on_context_overflow="final_turn")
    with pytest.raises(ValueError, match="final_turn_message"):
        ChainRollout(final_turn_on_max_turns=True)
    with pytest.raises(ValueError, match="final_turn_reserve"):
        ChainRollout(on_context_overflow="final_turn", final_turn_message=FINAL, final_turn_reserve=-1)
