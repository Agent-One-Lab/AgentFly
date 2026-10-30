"""The episode's Context: its main agent, strategy and sessions, and no loop data in metadata.

``context.metadata`` holds task data that tools also pass to environments (e.g. as
``env.reset`` arguments), and every conversation of the episode shares it, so the turn loop
must not write into it.
"""

import types

import pytest

from agentfly.agents.rollout.loop import generation
from agentfly.agents.rollout.sessions import Session
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
    chain = Chain({"episode_id": "e1"})
    messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    context = Context(rollout_id="e1")
    if folded:
        summarize_turn(chain, messages)
        assert context.trajectory_format == "flat"  # folding no longer writes to the context
    seen = {}

    async def fake_reward(agent, ctx, *, trajectory, **kwargs):
        seen["format"], seen["trajectory"] = ctx.trajectory_format, trajectory
        return None

    async def no_release(ctx):
        return None

    monkeypatch.setattr(strategy, "compute_reward", fake_reward)
    monkeypatch.setattr(strategy, "release_resources", no_release)
    current = Step(type="Action Input", messages=Messages.from_turns(messages))
    await strategy.finalize_chain(None, "e1", chain, current, 1, context)

    assert seen["format"] == ("segmented" if folded else "flat")
    assert seen["trajectory"] == (chain.histories if folded else current.messages.messages)
