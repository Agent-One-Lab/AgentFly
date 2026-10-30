"""The strategies' turn loops (``run_loop``) run only the turns.

They take the agent, context and loop state as arguments: no context set-up, reward,
resource release or metrics, and no rollout-level agent. ChainRollout's loop can be
called again on the same state to continue a conversation.
"""

import pytest
from test_rollout_segments import FakeAgent, fake_generation, isolated_runtime  # noqa: F401

from agentfly.agents.rollout import base as rollout_base
from agentfly.agents.rollout.events import FinishReason, MessageProduced, ToolObserved
from agentfly.agents.rollout.loop import metrics as rollout_metrics
from agentfly.agents.rollout.strategies import chain_rollout
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout, ChainState
from agentfly.agents.rollout.strategies.step_rollout import StepRollout, WindowState
from agentfly.agents.rollout.structures import Chain


class StrictContext:
    """A context the loop may use for tools, but must never release."""

    def __init__(self):
        self.metadata = {}
        self.trajectory_format = "flat"

    async def release_resource(self, **kwargs):
        pytest.fail("run_loop must not release resources")

    async def end_resource(self, **kwargs):
        pytest.fail("run_loop must not end resources")


@pytest.fixture
def loop_only(monkeypatch, isolated_runtime):  # noqa: F811
    """Fail on anything outside the turn loop."""

    async def no_reward(*args, **kwargs):
        pytest.fail("run_loop must not compute the reward")

    def no_context(**kwargs):
        pytest.fail("run_loop must not create a context")

    def no_episode_metrics(*args, **kwargs):
        pytest.fail("run_loop must not record episode metrics")

    monkeypatch.setattr(rollout_base, "calculate_reward", no_reward)
    monkeypatch.setattr(rollout_base, "Context", no_context)
    monkeypatch.setattr(rollout_metrics.RolloutMetrics, "record_episode", no_episode_metrics)
    return isolated_runtime


def new_chain_state(episode_id="e1"):
    chain = Chain({"episode_id": episode_id})
    root = chain.add_step(type="Action Input", messages=[
        {"role": "system", "content": "SYS"}, {"role": "user", "content": "O0"},
    ])
    return ChainState(episode_id=episode_id, info=chain.info, chain=chain, current_step=root)


async def drain(gen):
    return [event async for event in gen]


@pytest.mark.asyncio
async def test_chain_loop_runs_only_the_turns(loop_only):
    strategy = ChainRollout()
    agent = FakeAgent([["act"], ["act"]])
    state = new_chain_state()
    events = await drain(strategy.run_loop(
        agent=agent, context=StrictContext(), state=state, tools=[], max_turns=2, generation_config={},
    ))
    assert state.depth == 2
    assert agent.tool_calls == 2
    assert state.chain.info["finish_reason"] == FinishReason.MAX_TURNS
    assert sum(isinstance(e, MessageProduced) for e in events) == 2
    assert strategy.trajectories == {}
    assert not hasattr(strategy, "agent")


@pytest.mark.asyncio
async def test_chain_loop_continues_the_same_conversation(loop_only, monkeypatch):
    prompts = []

    async def recording_generation(agent, chain, current_step, *args, **kwargs):
        prompts.append([m["content"] for m in current_step.messages.messages])
        return await fake_generation(agent, chain, current_step, *args, **kwargs)

    monkeypatch.setattr(chain_rollout, "generate_response", recording_generation)
    strategy = ChainRollout()
    agent = FakeAgent([["act"]] * 4)
    state = new_chain_state()
    context = StrictContext()

    await drain(strategy.run_loop(
        agent=agent, context=context, state=state, tools=[], max_turns=2, generation_config={},
    ))
    assert state.depth == 2

    # Continue with a new user message, as a follow-up question would.
    state.chain.info.pop("finish_reason")
    state.current_step.messages.append({"role": "user", "content": "Q2"})
    await drain(strategy.run_loop(
        agent=agent, context=context, state=state, tools=[], max_turns=2, generation_config={},
    ))
    assert state.depth == 4
    assert len(prompts) == 4
    third_prompt = str(prompts[2])
    for earlier in ("O0", "A1", "O1", "A2", "O2", "Q2"):
        assert earlier in third_prompt
    assert prompts[3][: len(prompts[2])] == prompts[2]  # still one growing conversation


@pytest.mark.asyncio
async def test_chain_loop_uses_the_agent_it_is_given(loop_only):
    strategy = ChainRollout()
    run_agent = FakeAgent([["act"]] * 3)
    other_agent = FakeAgent([["act"]] * 3)
    other_agent.max_model_len = chain_rollout.TOKEN_SAFETY_MARGIN + 15  # ends after one turn
    state = new_chain_state()
    await drain(strategy.run_loop(
        agent=other_agent, context=StrictContext(), state=state, tools=[], max_turns=3, generation_config={},
    ))
    assert run_agent.tool_calls == 0
    assert other_agent.tool_calls == 1
    assert state.chain.info["finish_reason"] == FinishReason.MAX_MODEL_LEN


@pytest.mark.asyncio
async def test_step_loop_runs_only_the_turns(loop_only):
    strategy = StepRollout()
    agent = FakeAgent([["act"], ["act"], ["act"]], terminal_after=2)
    state = WindowState(
        episode_id="e1",
        info={},
        current={"raw_obs": "O0", "obs": "O0", "admissible": [], "task_description": ""},
        task_description="",
        full_messages=[{"role": "user", "content": "O0"}],
    )
    events = await drain(strategy.run_loop(
        agent=agent, context=StrictContext(), state=state, tools=[], max_turns=3, generation_config={},
    ))
    # The same events as ChainRollout's loop: one per generation and one per tool result.
    assert [type(e) for e in events] == [MessageProduced, ToolObserved] * 2
    assert [e.depth for e in events] == [0, 0, 1, 1]
    assert events[1].tool_name == "act" and events[1].observation == "O1"
    assert events[2].message is state.full_messages[3]
    assert agent.tool_calls == 2
    assert state.depth == 2
    assert len(state.steps) == 2
    assert state.finish_reason == FinishReason.TOOL_CONTROL_END
    assert [m["role"] for m in state.full_messages] == ["user", "assistant", "tool", "assistant", "tool"]
    assert len(state.history) == 1  # the ending turn is not added to the window
    assert not hasattr(strategy, "agent")


def test_strategies_keep_no_agent():
    assert not hasattr(ChainRollout(), "agent")
    assert not hasattr(StepRollout(), "agent")
