"""Subagents inside an episode: ``agent.run(..., context=...)`` starts or continues a session.

A tool of the main agent starts a subagent with ``subagent.run([brief], context=context)``
and asks it again with ``session_id=k``. The episode's strategy runs the session (no env
reset, reward or metrics); the episode's trajectory holds every session's segments, and
``Trajectory.sessions`` each session's stats.
"""

import pytest
from test_rollout_segments import FakeAgent, FakeContext, fake_generation, isolated_runtime  # noqa: F401

from agentfly.agents.agent_base import BaseAgent
from agentfly.agents.rollout.base import Rollout
from agentfly.agents.rollout.events import FinishReason
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.rollout.strategies.step_rollout import StepRollout
from agentfly.agents.utils.action_format import message_text
from agentfly.tools.types import ToolResult

SUB_TOOLS = [{"type": "function", "function": {"name": "look"}}]


class SubAgent(FakeAgent):
    """A scripted subagent: ``tool_names[depth]`` per turn, its own settings, and the
    real ``BaseAgent.run`` (so the nested path is the one tools use)."""

    max_model_len = 16384
    run = BaseAgent.run

    def __init__(self, name, tool_names):
        super().__init__(tool_names)
        self.name = name
        self.running_seen = []

    def prompt_tools(self):
        return SUB_TOOLS

    def _preprocess_messages(self, messages):
        return [{"messages": [{"role": "system", "content": f"SUB {self.name}"}, *messages]}]

    async def execute_tool_call(self, context, tool_call, *args):
        self.running_seen.append(list(context.running))
        return await super().execute_tool_call(context, tool_call, *args)


class MainAgent(FakeAgent):
    """A scripted main agent whose tools start and ask subagents through ``agent.run``."""

    def __init__(self, tool_names, subagents):
        super().__init__(tool_names)
        self.subagents = list(subagents)
        self.results = []
        self.running_seen = []

    async def execute_tool_call(self, context, tool_call, *args):
        self.tool_calls += 1
        name = tool_call["function"]["name"]
        self.running_seen.append(list(context.running))
        if name == "spawn":
            subagent = self.subagents.pop(0)
            result = await subagent.run([{"role": "user", "content": "brief"}], max_turns=2, context=context)
        elif name == "ask":
            subagent = context.sessions[1].agent
            result = await subagent.run([{"role": "user", "content": "Q2"}], max_turns=1, context=context, session_id=1)
        else:
            return ToolResult(name=name, arguments={}, observation="done", control="end")
        self.results.append(result)
        return ToolResult(name=name, arguments={}, observation=message_text(result.session.last_assistant))


def make_agents():
    sub_a = SubAgent("explorer", [["look"], [], []])  # run 1: look, stop; run 2: stop
    sub_b = SubAgent("checker", [["look"], []])
    main = MainAgent([["spawn"], ["spawn"], ["ask"], ["finish"]], [sub_a, sub_b])
    return main, sub_a, sub_b


async def run_episode(strategy, main):
    return await strategy.run(
        agent=main,
        messages=[{"messages": [{"role": "user", "content": "task"}], "task_id": "t"}],
        max_turns=len(main.tool_names),
        group_size=1,
    )


def record_contexts(strategy):
    contexts = []
    make_context = strategy.make_context

    def recording(**kwargs):
        context = make_context(**kwargs)
        contexts.append(context)
        return context

    strategy.make_context = recording
    return contexts


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [ChainRollout, StepRollout])
async def test_spawn_spawn_ask_finish(strategy_cls, isolated_runtime):
    strategy = strategy_cls()
    contexts = record_contexts(strategy)
    main, sub_a, sub_b = make_agents()
    trajectory = (await run_episode(strategy, main))[0]
    context = contexts[0]

    # Sessions: the main agent and two subagents it started, at its turns 0 and 1.
    assert sorted(context.sessions) == [0, 1, 2]
    assert [(s.parent_id, s.spawned_at, s.agent) for s in context.sessions.values()] == [
        (None, None, main),
        (0, 0, sub_a),
        (0, 1, sub_b),
    ]
    # The session acting is the last of context.running; nothing runs after the episode.
    assert main.running_seen == [[0], [0], [0], [0]]
    assert sub_a.running_seen == [[0, 1]] and sub_b.running_seen == [[0, 2]]
    assert context.running == []

    # Nested runs return no trajectories, and where the session stopped.
    assert [r.trajectories for r in main.results] == [[], [], []]
    assert [r.session.session_id for r in main.results] == [1, 2, 1]
    assert {r.rollout for r in main.results} == {strategy.name}
    assert main.results[0].session.finish_reason == FinishReason.NO_TOOL_CALLS

    # Subagent 1's conversation continues: its second run sees its first answer, then Q2.
    texts = [message_text(m) for m in strategy.transcript(context.sessions[1].state)[-1]]
    assert texts[0] == "SUB explorer"
    assert texts.index("A2") < texts.index("Q2") < texts.index("A3")

    # Segments: the main agent's, then each subagent's with its settings.
    by_session = [s.session_id for s in trajectory.segments]
    assert by_session == sorted(by_session) and set(by_session) == {0, 1, 2}
    for segment in trajectory.segments:
        if segment.session_id == 0:
            assert (segment.agent, segment.tools, segment.max_model_len) == (None, None, None)
        else:
            name = {1: "explorer", 2: "checker"}[segment.session_id]
            assert (segment.agent, segment.tools, segment.max_model_len) == (name, SUB_TOOLS, 16384)

    # Stats: the trajectory's own are the main agent's; sessions lists everyone's.
    assert trajectory.num_turns == 4
    assert trajectory.tool_call_counts == {"spawn": 2, "ask": 1, "finish": 1}
    assert [(e["session_id"], e["parent_id"], e["agent"], e["spawned_at"]) for e in trajectory.sessions] == [
        (0, None, None, None),
        (1, 0, "explorer", 0),
        (2, 0, "checker", 1),
    ]
    assert [e["turns"] for e in trajectory.sessions] == [4, 3, 2]
    assert [e["tool_call_counts"] for e in trajectory.sessions] == [
        trajectory.tool_call_counts,
        {"look": 1},
        {"look": 1},
    ]
    assert trajectory.sessions[0]["finish_reason"] == trajectory.finish_reason

    # Per-agent metrics from the outer run only.
    metrics = {e.name: e.value for e in isolated_runtime if e.kind == "scalar"}
    assert metrics["agent/rollout/avg_subagents"] == 2.0
    assert metrics["agent/rollout/explorer/avg_turns"] == 3.0
    assert metrics["agent/rollout/checker/avg_tool_calls"] == 1.0
    assert metrics["agent/rollout/avg_turns"] == 4.0


@pytest.mark.asyncio
async def test_reward_and_release_happen_once(isolated_runtime, monkeypatch):
    from agentfly.agents.rollout import base as rollout_base

    rewards, releases = [], []
    calculate_reward = rollout_base.calculate_reward

    async def counting_reward(*args, **kwargs):
        rewards.append(kwargs.get("trajectory"))
        return await calculate_reward(*args, **kwargs)

    monkeypatch.setattr(rollout_base, "calculate_reward", counting_reward)
    strategy = ChainRollout()

    async def counting_release(context):
        releases.append(context)

    strategy.release_resources = counting_release
    main, _, _ = make_agents()
    await run_episode(strategy, main)
    assert len(rewards) == 1 and len(releases) == 1
    # The reward reads the main agent's conversation.
    assert "SUB explorer" not in [message_text(m) for m in rewards[0]]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [ChainRollout, StepRollout])
async def test_without_subagents_nothing_new_is_logged(strategy_cls, isolated_runtime):
    strategy = strategy_cls()
    main = FakeAgent([["look"], ["look"]])
    trajectory = (await run_episode(strategy, main))[0]
    assert {s.session_id for s in trajectory.segments} == {0}
    assert len(trajectory.sessions) == 1
    entry = trajectory.sessions[0]
    assert (entry["turns"], entry["tool_call_counts"]) == (trajectory.num_turns, trajectory.tool_call_counts)
    names = {e.name for e in isolated_runtime}
    assert "agent/rollout/avg_subagents" not in names
    assert not any(name.startswith("agent/rollout/subagent/") for name in names)


def episode_with_main_session(strategy):
    context = FakeContext(metadata={}, rollout_id="e1")
    context.rollout = strategy
    main = FakeAgent([["look"]])
    strategy.add_session(context, main, [{"role": "user", "content": "task"}], info={"episode_id": "e1"})
    return context, main


@pytest.mark.asyncio
async def test_errors(isolated_runtime):
    strategy = ChainRollout()
    context, main = episode_with_main_session(strategy)
    sub = SubAgent("explorer", [["look"], [], []])
    brief = [{"role": "user", "content": "brief"}]

    with pytest.raises(RuntimeError, match="No session is running"):
        await sub.run(brief, max_turns=1, context=context)

    context.running.append(0)
    result = await sub.run(brief, max_turns=1, context=context)
    assert result.session.session_id == 1

    with pytest.raises(ValueError, match="No session 5"):
        await sub.run([{"role": "user", "content": "Q"}], max_turns=1, context=context, session_id=5)
    with pytest.raises(ValueError, match="Session 0 is the episode's main agent"):
        await sub.run([{"role": "user", "content": "Q"}], max_turns=1, context=context, session_id=0)
    other = SubAgent("explorer", [[]])
    with pytest.raises(ValueError, match="run by its own agent"):
        await other.run([{"role": "user", "content": "Q"}], max_turns=1, context=context, session_id=1)
    context.running.append(1)
    with pytest.raises(RuntimeError, match="Session 1 is running"):
        await sub.run([{"role": "user", "content": "Q"}], max_turns=1, context=context, session_id=1)
    context.running.pop()

    with pytest.raises(ValueError, match="takes one message"):
        await sub.run(
            [{"role": "user", "content": "Q"}, {"role": "user", "content": "R"}],
            max_turns=1,
            context=context,
            session_id=1,
        )
    with pytest.raises(ValueError, match="one conversation"):
        await sub.run([{"messages": brief}, {"messages": brief}], max_turns=1, context=context, session_id=1)
    with pytest.raises(TypeError, match="no rollout arguments"):
        await sub.run(brief, max_turns=1, context=context, group_size=2)
    with pytest.raises(ValueError, match="session_id needs a context"):
        await sub.run(brief, max_turns=1, session_id=1)


def test_a_strategy_without_add_message_does_not_support_subagents():
    with pytest.raises(NotImplementedError, match="does not support subagents"):
        Rollout().add_message(None, None, {"role": "user", "content": "Q"})


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["tool", "no_tool_call"])
async def test_step_continuing_puts_the_ending_turn_in_the_window(ending, isolated_runtime):
    strategy = StepRollout()
    context, _ = episode_with_main_session(strategy)
    agent = FakeAgent([["look"] if ending == "tool" else []], terminal_after=1)
    state = strategy.new_state(agent, [{"role": "user", "content": "brief"}], episode_id="e1", info={})
    async for _ in strategy.run_loop(
        agent=agent, context=context, state=state, tools=[], max_turns=1, generation_config=None
    ):
        pass
    assert state.history == [] and state.pending_turn is not None and state.finish_reason is not None

    strategy.add_message(agent, state, {"role": "user", "content": "Q2"})
    assert [turn["response"] for turn in state.history] == ["A1"]
    assert state.history[0]["obs"] == "brief"
    assert (state.current["obs"], state.pending_turn, state.finish_reason) == ("Q2", None, None)
    assert message_text(state.full_messages[-1]) == "Q2"


def test_chain_continuing_adds_a_step_after_the_current_one(isolated_runtime):
    strategy = ChainRollout()
    agent = FakeAgent([[]])
    state = strategy.new_state(agent, [{"role": "user", "content": "brief"}], episode_id="e1", info={})
    state.current_step.total_token_length = 100
    state.current_step.is_terminal = True
    state.info["finish_reason"] = FinishReason.NO_TOOL_CALLS

    strategy.add_message(agent, state, {"role": "user", "content": "Q2"})
    step = state.current_step
    assert not step.is_terminal and "finish_reason" not in state.info
    assert [message_text(m) for m in step.messages.messages] == ["brief", "Q2"]
    assert step.total_token_length == 100 + 1 + 15  # previous + observation + prefix
    assert state.chain.steps()[-1] is step
    assert state.depth == 1  # the ending turn's number is not reused
