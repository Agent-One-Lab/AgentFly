"""mini-swe-agent format-error semantics for tool-less turns under
``on_no_tool_call="continue"`` in the chain rollout: the corrective user nudge,
the consecutive-miss streak (reset by any tool-carrying turn), the strike limit,
the length-aware nudge variant, and the silent/unlimited configuration.

Reuses the isolated runtime (fake generation / context / reward) from
``test_rollout_segments``.
"""
import pytest

from agentfly.agents.rollout import base as rollout_base
from agentfly.agents.rollout.strategies import chain_rollout
from agentfly.agents.rollout.events import FinishReason

from test_rollout_segments import FakeAgent, isolated_runtime, roles  # noqa: F401


async def run(strategy, agent, *, max_turns=None, generation_config=None):
    return await strategy.run(
        agent=agent,
        messages=[{
            "messages": [{"role": "system", "content": "SYS"},
                         {"role": "user", "content": "O0"}],
            "task_id": "task_1",
        }],
        max_turns=max_turns if max_turns is not None else len(agent.tool_names),
        num_chains=1,
        generation_config=generation_config or {},
    )


def user_texts(segment):
    return [
        m["content"][0]["text"] if isinstance(m["content"], list) else m["content"]
        for m in segment.messages if m["role"] == "user"
    ]


# ---- construction / validation ------------------------------------------------

def test_defaults_match_miniswe():
    r = chain_rollout.ChainRollout()
    assert r.max_consecutive_no_tool_calls == 3
    assert r.no_tool_call_message == rollout_base.NO_TOOL_CALL_MESSAGE_MINISWE
    assert r.render_no_tool_call_message() == rollout_base.MINISWE_NO_TOOL_CALL_TEXT
    assert (
        r.render_no_tool_call_message(truncated=True)
        == rollout_base.MINISWE_NO_TOOL_CALL_TRUNCATED_TEXT
    )


def test_custom_and_disabled_messages():
    assert chain_rollout.ChainRollout(no_tool_call_message="say bash").render_no_tool_call_message(
        truncated=True
    ) == "say bash"
    assert chain_rollout.ChainRollout(no_tool_call_message=None).render_no_tool_call_message() is None
    assert chain_rollout.ChainRollout(no_tool_call_message="").render_no_tool_call_message() is None


def test_invalid_knobs_rejected():
    with pytest.raises(ValueError):
        chain_rollout.ChainRollout(max_consecutive_no_tool_calls=-1)
    with pytest.raises(ValueError):
        chain_rollout.ChainRollout(max_consecutive_no_tool_calls="three")
    with pytest.raises(ValueError):
        chain_rollout.ChainRollout(no_tool_call_message=3)


# ---- loop behaviour -------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_end_policy_is_unchanged(isolated_runtime):
    result = await run(chain_rollout.ChainRollout(), FakeAgent([[]]))
    assert result[0].finish_reason == FinishReason.NO_TOOL_CALLS
    assert roles(result[0].segments[0]) == ["system", "user", "assistant"]


@pytest.mark.asyncio
async def test_continue_nudges_then_recovers(isolated_runtime):
    strategy = chain_rollout.ChainRollout(on_no_tool_call="continue")
    result = await run(strategy, FakeAgent([[], ["t"]]))
    traj = result[0]
    segment = traj.segments[0]
    assert roles(segment) == ["system", "user", "assistant", "user", "assistant", "tool"]
    assert user_texts(segment)[1] == rollout_base.MINISWE_NO_TOOL_CALL_TEXT
    assert traj.finish_reason == FinishReason.MAX_TURNS
    assert traj.num_turns == 2  # the nudge is not a model turn
    assert traj.metadata["no_tool_call_nudges"] == 1
    assert traj.metadata["no_tool_call_turns"] == 1
    assert traj.metadata["no_tool_call_streak"] == 0  # reset by the tool turn


@pytest.mark.asyncio
async def test_three_consecutive_misses_end_the_chain(isolated_runtime):
    strategy = chain_rollout.ChainRollout(on_no_tool_call="continue")
    result = await run(strategy, FakeAgent([[], [], [], ["t"]]))
    traj = result[0]
    assert traj.finish_reason == FinishReason.REPEATED_NO_TOOL_CALLS
    assert traj.num_turns == 3
    # Two nudges: the third miss ends the chain before another nudge.
    assert traj.metadata["no_tool_call_nudges"] == 2
    assert traj.metadata["no_tool_call_streak"] == 3
    assert roles(traj.segments[-1]) == [
        "system", "user", "assistant", "user", "assistant", "user", "assistant",
    ]


@pytest.mark.asyncio
async def test_tool_turn_resets_the_streak(isolated_runtime):
    strategy = chain_rollout.ChainRollout(
        on_no_tool_call="continue", max_consecutive_no_tool_calls=2,
    )
    # miss, tool (reset), miss, miss -> ends on the second consecutive miss.
    result = await run(strategy, FakeAgent([[], ["t"], [], [], ["t"]]))
    traj = result[0]
    assert traj.finish_reason == FinishReason.REPEATED_NO_TOOL_CALLS
    assert traj.num_turns == 4
    assert traj.metadata["no_tool_call_turns"] == 3
    assert traj.metadata["no_tool_call_nudges"] == 2


@pytest.mark.asyncio
async def test_unlimited_and_silent_continue(isolated_runtime):
    strategy = chain_rollout.ChainRollout(
        on_no_tool_call="continue", max_consecutive_no_tool_calls=0, no_tool_call_message=None,
    )
    result = await run(strategy, FakeAgent([[], [], []]))
    traj = result[0]
    assert traj.finish_reason == FinishReason.MAX_TURNS
    assert roles(traj.segments[0]) == ["system", "user", "assistant", "assistant", "assistant"]
    assert "no_tool_call_nudges" not in traj.metadata
    assert traj.metadata["no_tool_call_turns"] == 3


@pytest.mark.asyncio
async def test_budget_filling_turn_gets_length_variant(isolated_runtime):
    strategy = chain_rollout.ChainRollout(on_no_tool_call="continue")
    # fake_generation returns two token ids per turn; a two-token budget means
    # the sampled response filled it (mini-swe's finish_reason == "length").
    result = await run(
        strategy, FakeAgent([[], ["t"]]), generation_config={"max_tokens": 2},
    )
    assert user_texts(result[0].segments[0])[1] == rollout_base.MINISWE_NO_TOOL_CALL_TRUNCATED_TEXT
    # Both turns filled the two-token budget, so both count as capped.
    assert result[0].metadata["capped_turns"] == 2


@pytest.mark.asyncio
async def test_larger_budget_gets_generic_variant(isolated_runtime):
    strategy = chain_rollout.ChainRollout(on_no_tool_call="continue")
    result = await run(
        strategy, FakeAgent([[], ["t"]]), generation_config={"max_tokens": 64},
    )
    assert user_texts(result[0].segments[0])[1] == rollout_base.MINISWE_NO_TOOL_CALL_TEXT
    assert "capped_turns" not in result[0].metadata


@pytest.mark.asyncio
async def test_nudge_counts_toward_context_limit(isolated_runtime, monkeypatch):
    # observation_token_length is patched to 1 by the fixture; make the nudge
    # push the running total over the ceiling so the chain ends on MAX_MODEL_LEN.
    monkeypatch.setattr(chain_rollout, "observation_token_length", lambda *a, **k: 10_000)
    agent = FakeAgent([[], ["t"]])
    agent.max_model_len = 5_000
    strategy = chain_rollout.ChainRollout(on_no_tool_call="continue")
    result = await run(strategy, agent)
    traj = result[0]
    assert traj.finish_reason == FinishReason.MAX_MODEL_LEN
    assert roles(traj.segments[-1]) == ["system", "user", "assistant", "user"]
