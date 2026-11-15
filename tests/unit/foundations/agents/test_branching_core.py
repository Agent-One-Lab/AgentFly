"""Branching, step 1 and 2 of the plan: environment snapshots, the Context branch-point record, the tree fields on a
Trajectory, and forced (replayed) responses in the generation step."""

import asyncio

import pytest

from agentfly.agents.rollout.loop import generation
from agentfly.agents.rollout.loop.generation import FORCED, generate_response
from agentfly.agents.rollout.structures import Chain
from agentfly.agents.types import Segment, Trajectory
from agentfly.core.context import Context
from agentfly.envs.env_base import BaseEnv


class CounterEnv(BaseEnv):
    def __init__(self):
        self.count = 0
        self.items = {"a": [1, 2]}

    async def start(self): pass
    async def reset(self): self.__init__()
    async def step(self, action): self.count += 1; return str(self.count)
    async def aclose(self): pass


def test_env_snapshot_and_restore_default_deep_copy():
    env = CounterEnv()
    asyncio.run(env.step("x"))
    snap = env.snapshot()
    asyncio.run(env.step("y")); env.items["a"].append(3)
    assert env.count == 2 and env.items["a"] == [1, 2, 3]
    env.restore(snap)
    assert env.count == 1 and env.items["a"] == [1, 2]
    env.items["a"].append(9)  # the restored state is a copy: the snapshot is not shared
    assert snap["items"]["a"] == [1, 2]


def test_context_records_one_branch_point_and_exposes_a_snapshot_to_restore():
    ctx = Context("ep", metadata={"task": 1})
    assert ctx.branch_point is None and ctx.env_snapshot is None
    assert ctx.record_branch_point({"world": 1}, 3, {"turn": 4}) is True
    assert ctx.record_branch_point({"world": 2}, 5) is False  # one per episode
    assert ctx.branch_point == {"snapshot": {"world": 1}, "session_id": 3, "payload": {"turn": 4}}
    branch = Context("ep2", metadata={"task": 1, "env_snapshot": {"world": 1}})
    assert branch.env_snapshot == {"world": 1}


def test_trajectory_tree_fields_default_to_a_one_leaf_tree():
    t = Trajectory(segments=[Segment(messages=[])], episode_id="e1")
    assert (t.tree_id, t.leaf_id, t.branch_of, t.branch_turn, t.branch_point) == (None, 0, None, None, None)
    b = Trajectory(segments=[Segment(messages=[])], episode_id="e2", tree_id="e1", leaf_id=1, branch_of="e1",
                   branch_turn={"session_id": 0, "turn": 2})
    assert b.branch_of == "e1" and b.branch_turn["turn"] == 2


class ScriptedAgent:
    """generate_async must not be called while a forced response is pending."""
    tokenizer = None
    default_tool_call_source = "parser"

    def __init__(self):
        self.calls = 0

    def skills_payload(self):
        return None

    async def generate_async(self, messages_list, **kwargs):
        self.calls += 1
        return {"response_texts": ["sampled"]}

    def parse(self, response_texts, tool_calls=None, context=None, trajectory_segments=None):
        return [{"role": "assistant", "content": text} for text in response_texts]


@pytest.mark.asyncio
async def test_forced_responses_are_replayed_once_then_sampling_resumes(monkeypatch):
    monkeypatch.setattr(generation, "prepare_generation_config", lambda agent, cfg, step, tools=None: {})
    agent = ScriptedAgent()
    chain = Chain(info={})
    step = chain.add_step(type="Action Input", messages=[{"role": "user", "content": "go"}])
    chain.forced_responses = ["<delegate>{\"goal\": \"x\"}</delegate>"]
    msg, _ = await generate_response(agent, chain, step, [], 0, "ep", {}, Context("ep"))
    assert msg["content"] == "<delegate>{\"goal\": \"x\"}</delegate>" and msg.get(FORCED) is True
    assert agent.calls == 0 and chain.forced_responses == []
    msg2, _ = await generate_response(agent, chain, step, [], 1, "ep", {}, Context("ep"))
    assert msg2["content"] == "sampled" and FORCED not in msg2 and agent.calls == 1
