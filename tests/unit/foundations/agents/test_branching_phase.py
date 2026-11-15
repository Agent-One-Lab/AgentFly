"""Branching, step 3: a tool records a branch point; after the main episodes the rollout runs a second leaf from the
environment snapshot with the recorded turn replayed; both leaves share a tree and the branch's prefix is context."""

import pytest

from agentfly.agents.rollout import base as rollout_base
from agentfly.agents.rollout.loop import metrics as rollout_metrics
from agentfly.agents.rollout.strategies import chain_rollout
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.utils.action_format import message_text
from agentfly.rewards.types import RewardResult
from agentfly.tools.types import ToolResult


class World:
    """A toy environment: a counter the tools advance; snapshot/restore via the BaseEnv defaults."""

    def __init__(self):
        self.count = 0

    def snapshot(self):
        return {"count": self.count}

    def restore(self, state):
        self.count = state["count"]


class BranchingAgent:
    """Turn 1: act; turn 2: delegate (the tool records the branch point); later turns: act. The world is one
    object per context, restored from the snapshot in a branch context."""
    system_prompt = "SYS"
    tokenizer = None
    processor = None
    max_model_len = None
    reward_fn = object()
    name = "main"

    def __init__(self):
        self.worlds = {}
        self.sampled = []  # (episode, depth) pairs that were really sampled

    def prompt_tools(self):
        return []

    def maybe_append_context_trigger_user_message(self, step):
        pass

    def extract_final_response(self, messages):
        return "answer"

    def world(self, context):
        if context.rollout_id not in self.worlds:
            w = World()
            if context.env_snapshot is not None:
                w.restore(context.env_snapshot)
            self.worlds[context.rollout_id] = w
        return self.worlds[context.rollout_id]

    async def execute_tool_call(self, context, tool_call, *args):
        w = self.world(context)
        name = tool_call["function"]["name"]
        if name == "delegate":
            context.record_branch_point(w.snapshot(), 0, {"goal": "part"})
            w.count += 10  # the child's work
            return ToolResult(name=name, arguments={}, observation=f"child done, count={w.count}")
        w.count += 1
        return ToolResult(name=name, arguments={}, observation=f"count={w.count}", control="end" if w.count >= 13 else None)


async def scripted_generation(agent, chain, current_step, tools, depth, episode_id, generation_config, context, use_token_ids=True):
    forced = chain.forced_responses
    if forced:
        text = forced.pop(0)
        msg = {"role": "assistant", "content": text, "forced": True}
    else:
        agent.sampled.append((episode_id, depth))
        text = "A%d" % (depth + 1)
        msg = {"role": "assistant", "content": text}
    name = "delegate" if text == "A2" else "act"
    msg["tool_calls"] = [{"id": "c", "type": "function", "function": {"name": name, "arguments": "{}"}}]
    return msg, 10


async def fake_reward(*args, **kwargs):
    return RewardResult(reward=1.0, metrics={})


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(chain_rollout, "generate_response", scripted_generation)
    monkeypatch.setattr(rollout_base, "calculate_reward", fake_reward)
    monkeypatch.setattr(chain_rollout, "observation_token_length", lambda *a, **k: 1)
    monkeypatch.setattr(chain_rollout, "estimate_chat_prompt_tokens", lambda *a, **k: 10)
    monkeypatch.setattr(rollout_base, "progress_bar", lambda *a, **k: None)
    monkeypatch.setattr(rollout_base.Monitor, "ensure_started", lambda: None)
    monkeypatch.setattr(rollout_metrics, "emit", lambda event: None)


@pytest.mark.asyncio
async def test_branch_phase_runs_a_second_leaf_from_the_snapshot(runtime):
    agent = BranchingAgent()
    result = await ChainRollout().run(
        agent=agent,
        messages=[{"messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "O0"}], "task_id": "t"}],
        max_turns=6,
        group_size=1,
    )
    trajectories = result.trajectories
    assert len(trajectories) == 2  # the episode and its branch
    source, branch = trajectories
    assert source.branch_point == {"session_id": 0, "turn": 1, "payload": {"goal": "part"}}
    assert source.tree_id == source.episode_id and source.leaf_id == 0 and source.branch_of is None
    assert branch.group_id == source.group_id and branch.episode_idx == 1
    assert branch.tree_id == source.episode_id and branch.leaf_id == 1 and branch.branch_of == source.episode_id
    assert branch.branch_turn == {"session_id": 0, "turn": 1}
    # the branch replayed the delegate turn (not sampled) and then sampled only its remaining turns
    sampled_in_branch = [d for e, d in agent.sampled if e == branch.episode_id]
    assert sampled_in_branch and min(sampled_in_branch) == 2
    assistant = [m for m in branch.segments[0].messages if m.get("role") == "assistant"]
    assert message_text(assistant[0]) == "A1" and "forced" not in assistant[0]  # copied from the source
    assert message_text(assistant[1]) == "A2" and assistant[1].get("forced") is True  # replayed
    assert branch.segments[0].trained_from_turn == 1  # the first turn is context; the replayed one is masked too
    # the world was restored: the branch's child worked on the pre-delegation count (1 + 10), like the source's
    tool_texts = [message_text(m) for m in branch.segments[0].messages if m.get("role") == "tool"]
    assert any("child done, count=11" in t for t in tool_texts)
    # both leaves end as full episodes with rewards
    assert source.reward == 1.0 and branch.reward == 1.0
    assert branch.num_turns == source.num_turns - 1  # the branch's own turns: the replayed one and the sampled rest


@pytest.mark.asyncio
async def test_branch_flag_off_runs_no_second_leaf(runtime):
    agent = BranchingAgent()
    result = await ChainRollout(branch=False).run(
        agent=agent,
        messages=[{"messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "O0"}], "task_id": "t"}],
        max_turns=6,
        group_size=1,
    )
    assert len(result.trajectories) == 1
    assert result.trajectories[0].branch_point == {"session_id": 0, "turn": 1, "payload": {"goal": "part"}}
