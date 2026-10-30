"""The episode's Context: its main agent, strategy and sessions, and no loop data in metadata.

``context.metadata`` holds task data that tools also pass to environments (e.g. as
``env.reset`` arguments), and every conversation of the episode shares it, so the turn loop
must not write into it. Every episode runs its main agent as session 0, created through
the strategy contract.
"""

import types

import pytest
from test_rollout_segments import FakeAgent, fake_generation, isolated_runtime  # noqa: F401

from agentfly.agents.rollout.events import EpisodeEnded, EpisodeStarted, MessageProduced, ToolObserved
from agentfly.agents.rollout.loop import generation
from agentfly.agents.rollout.sessions import Session, SessionResult
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.rollout.strategies.step_rollout import StepRollout
from agentfly.agents.rollout.structures import Chain, Step
from agentfly.agents.utils.messages import Messages
from agentfly.core import Context


def test_a_plain_context_has_no_agent_strategy_or_sessions():
    context = Context(rollout_id="demo")
    assert context.agent is None and context.rollout is None and context.sessions == {}


@pytest.mark.parametrize("strategy_cls", [ChainRollout, StepRollout])
def test_make_context_records_the_main_agent_and_the_strategy(strategy_cls):
    strategy, agent = strategy_cls(), object()
    context = strategy.make_context(agent=agent, episode_id="e1", group_id="g1", metadata={"task_id": "t"})
    assert context.agent is agent and context.rollout is strategy
    assert (context.rollout_id, context.group_id, context.metadata) == ("e1", "g1", {"task_id": "t"})
    assert context.sessions == {}


def test_session_is_plain_data():
    session = Session(id=1, parent_id=0, agent="A", state="S")
    assert (session.spawned_at, session.info, session.task) == (None, {}, None)


class RecordingAgent:
    """Just enough of an agent for ``generate_response``; records what ``parse`` receives."""

    max_model_len = None
    default_tool_call_source = "parser"

    def __init__(self):
        self.parse_kwargs = None

    def skills_payload(self):
        return None

    async def generate_async(self, messages_list, **kwargs):
        return {"response_texts": ["<action>look</action>"]}

    def parse(self, responses, tool_calls=None, context=None, **kwargs):
        self.parse_kwargs = kwargs
        return [{"role": "assistant", "content": [{"type": "text", "text": responses[0]}]}]


@pytest.mark.asyncio
async def test_generation_passes_the_prompt_to_the_parser_and_leaves_metadata_alone():
    agent = RecordingAgent()
    context = Context(rollout_id="e1", metadata={"task_id": "t"})
    metadata = context.metadata
    prompt = [{"role": "user", "content": "O0"}]
    step = Step(type="Action Input", messages=Messages.from_turns(list(prompt)))

    message, _ = await generation.generate_response(agent, Chain({}), step, [], 0, "e1", {}, context)

    assert message["content"][0]["text"] == "<action>look</action>"
    assert agent.parse_kwargs["trajectory_segments"] == [step.messages.messages]
    assert context.metadata is metadata and context.metadata == {"task_id": "t"}


def test_action_agent_parse_reads_the_prompt_from_its_argument():
    from agentfly.agents.specialized.action_agent import ActionAgent

    agent = types.SimpleNamespace(action_tool_name="act")
    seen = []
    agent._parse_single_response = lambda response, segment: seen.append(segment) or {"text": response}
    ActionAgent.parse(agent, ["r0", "r1"], trajectory_segments=[["p0"]])
    assert seen == [["p0"], []]  # a response without a prompt gets an empty one
    seen.clear()
    ActionAgent.parse(agent, ["r0"], context=Context(rollout_id="e", metadata={"trajectory_segments": [["old"]]}))
    assert seen == [[]]  # metadata is no longer read


def test_qwen3coder_parse_sees_the_trigger_prompt_from_its_argument(monkeypatch):
    from agentfly.agents.agent_base import BaseAgent
    from agentfly.agents.specialized.swe_agents.qwen3coder_swe_agent import Qwen3CoderSWEAgent

    monkeypatch.setattr(BaseAgent, "parse", lambda self, responses, **kwargs: [{"plain": r} for r in responses])
    agent = object.__new__(Qwen3CoderSWEAgent)
    agent.context_trigger_message = "SUMMARIZE NOW"
    triggered = [{"role": "user", "content": "SUMMARIZE NOW"}]

    forced, = agent.parse(["the summary"], trajectory_segments=[triggered])
    assert forced["tool_calls"][0]["function"]["name"] == "summarize"  # the forced summary turn
    plain, = agent.parse(["the summary"], context=Context(rollout_id="e", metadata={"trajectory_segments": [triggered]}))
    assert plain == {"plain": "the summary"}  # metadata is no longer read


def summarize_turn(chain, messages_before, summary="S"):
    """Fold the conversation the way a summarize tool result does."""
    strategy = ChainRollout()
    return strategy.apply_context_folding(
        None, chain=chain, messages=Messages.from_turns(messages_before), tool_name="summarize",
        observation=summary, tool_call_id="c1", tool_result_name="summarize", new_content=[],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("folded", [False, True])
async def test_reward_sees_the_trajectory_format_set_just_before_it(monkeypatch, folded):
    strategy = ChainRollout()
    messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    state = strategy.new_state(None, messages, episode_id="e1", info={"episode_id": "e1"})
    context = Context(rollout_id="e1")
    if folded:
        summarize_turn(state.chain, messages)
        assert context.trajectory_format == "flat"  # folding no longer writes to the context
    seen = {}

    async def fake_reward(agent, ctx, *, trajectory, **kwargs):
        seen["format"], seen["trajectory"] = ctx.trajectory_format, trajectory
        return None

    async def no_release(ctx):
        return None

    monkeypatch.setattr(strategy, "compute_reward", fake_reward)
    monkeypatch.setattr(strategy, "release_resources", no_release)
    monkeypatch.setattr(strategy.metrics, "record_episode", lambda **kwargs: None)
    await strategy.finish_episode(None, context, state, started_at=0.0)

    current = state.current_step.messages.messages
    assert seen["format"] == ("segmented" if folded else "flat")
    assert seen["trajectory"] == ([*state.chain.histories, current] if folded else current)


# ---- session 0 and the strategy contract ----


def record_contexts(strategy):
    """The contexts the strategy makes, in order."""
    contexts = []
    make_context = strategy.make_context

    def recording(**kwargs):
        context = make_context(**kwargs)
        contexts.append(context)
        return context

    strategy.make_context = recording
    return contexts


async def run_two_turns(strategy, agent):
    return await strategy.run(
        agent=agent,
        messages=[{"messages": [{"role": "user", "content": "O0"}], "task_id": "t"}],
        max_turns=2,
        group_size=2,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [ChainRollout, StepRollout])
async def test_each_episode_runs_its_main_agent_as_session_0(strategy_cls, isolated_runtime):
    strategy = strategy_cls()
    contexts = record_contexts(strategy)
    agent = FakeAgent([["act"], ["act"]])
    result = await run_two_turns(strategy, agent)

    assert len(contexts) == 2
    for context in contexts:
        assert list(context.sessions) == [0]
        session = context.sessions[0]
        assert (session.agent, session.parent_id, session.spawned_at) == (agent, None, None)
        assert session.state.episode_id == context.rollout_id
        assert session.state.info["episode_id"] == context.rollout_id
        assert session.state.depth == 2  # the state the loop ran on
        trajectory = next(t for t in result if t.episode_id == context.rollout_id)
        assert strategy.segments(session.state) == trajectory.segments
        assert strategy.records(session.state) == trajectory.steps
        assert strategy.result(session.state).finish_reason == trajectory.finish_reason


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [ChainRollout, StepRollout])
async def test_result_reads_where_the_session_stopped(strategy_cls, isolated_runtime):
    strategy = strategy_cls()
    contexts = record_contexts(strategy)
    await run_two_turns(strategy, FakeAgent([["act"], ["act"]], terminal_after=2))
    state = contexts[0].sessions[0].state

    result = strategy.result(state)
    assert isinstance(result, SessionResult)
    assert (result.finish_reason, result.turns) == ("tool_control_end", 2)
    assert result.last_assistant["content"][0]["text"] == "A2"
    assert result.last_tool_result.observation == "O2"


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [ChainRollout, StepRollout])
async def test_transcript_is_the_whole_conversation(strategy_cls, isolated_runtime):
    strategy = strategy_cls()
    contexts = record_contexts(strategy)
    await run_two_turns(strategy, FakeAgent([["act"], ["act"]]))
    transcript = strategy.transcript(contexts[0].sessions[0].state)

    assert len(transcript) == 1  # nothing folded
    assert [m["role"] for m in transcript[0]] == ["user", "assistant", "tool", "assistant", "tool"]


def test_new_states_start_from_the_given_messages():
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q"}]
    info = {"episode_id": "e1", "task_description": "the task"}

    chain_state = ChainRollout().new_state(None, messages, episode_id="e1", info=info)
    assert chain_state.info is info and chain_state.chain.info is info
    assert chain_state.current_step is chain_state.chain.root
    assert [m["role"] for m in chain_state.current_step.messages.messages] == ["system", "user"]
    assert chain_state.depth == 0

    window = StepRollout().new_state(None, messages, episode_id="e1", info=info)
    assert window.info is info
    assert window.current == {"raw_obs": "", "obs": "Q", "admissible": [], "task_description": "the task"}
    assert window.full_messages == messages and window.full_messages is not messages
    assert window.depth == 0 and window.steps == []


def test_step_takes_the_env_start_from_the_first_step_hook():
    strategy = StepRollout()
    state = strategy.new_state(
        None, [{"role": "user", "content": "Q"}], episode_id="e1", info={"task_description": "dataset task"},
    )
    context = Context(rollout_id="e1")
    strategy.after_first_step_hook(state, context)  # nothing recorded: unchanged
    assert state.current == {"raw_obs": "", "obs": "Q", "admissible": [], "task_description": "dataset task"}

    context.metadata.update(_flat_initial_obs="You are in a kitchen. Your task is to: heat water",
                            _flat_initial_admissible=["look"])
    strategy.after_first_step_hook(state, context)
    assert state.current == {
        "raw_obs": "You are in a kitchen. Your task is to: heat water",
        "obs": "Q",
        "admissible": ["look"],
        "task_description": "heat water",
    }
    assert state.task_description == "heat water"


@pytest.mark.asyncio
async def test_step_rollout_streams_the_same_events_as_chain(isolated_runtime):
    strategy = StepRollout()
    events = [
        event
        async for event in strategy.stream(
            FakeAgent([["act"], ["act"]], terminal_after=2),
            [{"role": "user", "content": "O0"}],
            max_turns=2,
            group_size=1,
        )
    ]
    assert [type(e) for e in events] == [
        EpisodeStarted, MessageProduced, ToolObserved, MessageProduced, ToolObserved, EpisodeEnded,
    ]
    assert events[-1].finish_reason == "tool_control_end"
    assert len(strategy.trajectories) == 1
