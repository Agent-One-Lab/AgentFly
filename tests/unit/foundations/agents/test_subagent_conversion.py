"""Training conversion of episodes with subagent sessions.

Each row is tokenized with its own agent's settings (a subagent segment's prompt tools and
max_model_len, else the main agent's); rows with the same settings are tokenized together and the
groups are merged back in row order. Each row records its session and agent.
"""

import types

import torch

from agentfly.agents.rollout import conversion
from agentfly.agents.types import RunResult, Segment, Trajectory

MAIN_TOOLS = [{"type": "function", "function": {"name": "spawn_subagent"}}]
SUB_TOOLS = [{"type": "function", "function": {"name": "scienceworld_explorer"}}]


def turns(*texts):
    return [{"role": "user", "content": texts[0]}] + [{"role": "assistant", "content": t} for t in texts[1:]]


def recording_tokenize(calls, pad_token_id=9):
    """A fake tokenize hook: row i of a call has length 2 + (number of messages), so groups have
    different widths; token values are the row's message count."""

    def tokenize(agent, messages_list, **kwargs):
        calls.append((kwargs.get("tools"), kwargs.get("max_length"), [m[0]["content"] for m in messages_list]))
        lengths = [2 + len(m) for m in messages_list]
        width = max(lengths)
        ids = torch.full((len(lengths), width), pad_token_id)
        attention = torch.zeros(len(lengths), width, dtype=torch.long)
        labels = torch.full((len(lengths), width), -100)
        for i, (n, m) in enumerate(zip(lengths, messages_list)):
            ids[i, :n] = len(m)
            attention[i, :n] = 1
            labels[i, n - 1] = len(m)
        action = attention.clone()
        reward = torch.zeros_like(attention)
        reward[torch.arange(len(lengths)), torch.tensor(lengths) - 1] = 1
        return {
            "input_ids": ids,
            "attention_mask": attention,
            "labels": labels,
            "action_mask": action,
            "reward_mask": reward,
            "position_ids": torch.clip(torch.cumsum(attention, dim=-1) - 1, min=0),
        }

    return tokenize


def agent(**extra):
    return types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(pad_token_id=9),
        processor=None,
        max_model_len=4096,
        prompt_tools=lambda: MAIN_TOOLS,
        **extra,
    )


def episode_with_a_subagent():
    main = Segment(messages=turns("main", "spawn"))
    sub = Segment(
        messages=turns("brief", "look", "look", "return"),
        session_id=1,
        agent="explorer",
        tools=SUB_TOOLS,
        max_model_len=16384,
    )
    return RunResult(
        trajectories=[Trajectory(segments=[main, sub], reward=1.0, group_id="G1", episode_id="E1")], rollout="chain"
    )


def test_subagent_rows_are_tokenized_with_their_own_settings_and_merged_in_order():
    calls = []
    inputs, rows, repeat, pad = conversion.build_segment_batch(
        agent(), episode_with_a_subagent(), tokenize=recording_tokenize(calls)
    )
    assert calls == [(MAIN_TOOLS, 4096, ["main"]), (SUB_TOOLS, 16384, ["brief"])]
    assert rows == [(0, 0), (0, 1)] and pad == 0

    # The main row (2 messages, 4 tokens) is padded to the subagent row's width (4 messages, 6 tokens).
    assert inputs["input_ids"].tolist() == [[2, 2, 2, 2, 9, 9], [4] * 6]
    assert inputs["attention_mask"].tolist() == [[1, 1, 1, 1, 0, 0], [1] * 6]
    assert inputs["labels"][0].tolist() == [-100, -100, -100, 2, -100, -100]
    assert inputs["action_mask"][0, 4:].tolist() == [0, 0]
    assert inputs["position_ids"].tolist() == [[0, 1, 2, 3, 3, 3], list(range(6))]
    # Both rows carry the episode's outcome on their reward token, and say where they come from.
    assert inputs["rm_scores"].sum(dim=-1).tolist() == [1.0, 1.0]
    assert inputs["session_id"].tolist() == [0, 1]
    assert inputs["agent_name"].tolist() == ["", "explorer"]


def test_without_subagent_rows_the_tokenize_call_is_unchanged():
    calls = []
    result = RunResult(
        trajectories=[Trajectory(segments=[Segment(messages=turns("main", "a"))], reward=0.0, group_id="G1")],
        rollout="chain",
    )
    # The main agent's settings are not even looked up.
    plain_agent = types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=9), processor=None)
    inputs, *_ = conversion.build_segment_batch(plain_agent, result, tokenize=recording_tokenize(calls))
    assert calls == [(None, None, ["main"])]
    assert inputs["session_id"].tolist() == [0]


def test_subagent_rows_with_the_main_agents_settings_share_the_plain_call():
    calls = []
    result = episode_with_a_subagent()
    result.trajectories[0].segments[1].tools = MAIN_TOOLS
    result.trajectories[0].segments[1].max_model_len = 4096
    conversion.build_segment_batch(agent(), result, tokenize=recording_tokenize(calls))
    assert calls == [(None, None, ["main", "brief"])]  # one settings group: the agent's defaults
