"""Trajectory inspection needs recorded data, not a model or an agent instance."""

from copy import deepcopy

import pytest

from agentfly.agents.types import RunResult, Segment, Trajectory
from agentfly.agents.utils.inspection import print_trajectory


def test_print_trajectory_renders_text_and_observations(capsys):
    trajectory = Trajectory(segments=[Segment(messages=[
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": [{"type": "text", "text": "Question"}]},
        {"role": "assistant", "content": "Answer"},
        {"role": "tool", "content": [{"type": "text", "text": "Observation"}]},
    ])])

    assert print_trajectory(trajectory) is None
    assert capsys.readouterr().out == (
        "system: Be helpful.\n"
        "user: Question\n"
        "assistant: Answer\n"
        "tool: Observation\n"
    )


@pytest.mark.parametrize("image_part", [
    {"type": "image", "image": "data:image/png;base64,private-payload"},
    {"type": "image", "image_url": "https://example.com/private-image"},
    {"type": "image_url", "image_url": {"url": "https://example.com/private-image"}},
])
def test_print_trajectory_uses_image_placeholders(capsys, image_part):
    trajectory = Trajectory(segments=[Segment(messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Before "},
            image_part,
            {"type": "text", "text": " after"},
        ]},
    ])])

    print_trajectory(trajectory)
    assert capsys.readouterr().out == "user: Before [Image] after\n"


def test_print_trajectory_does_not_dump_other_multimodal_payloads(capsys):
    trajectory = Trajectory(segments=[Segment(messages=[
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "private-payload"}},
        ]},
    ])])

    print_trajectory(trajectory)
    assert capsys.readouterr().out == "user: [input_audio]\n"


@pytest.mark.parametrize("content_fields,expected_text", [
    ({}, ""),
    ({"content": None}, ""),
    ({"content": ""}, ""),
    ({"content": []}, ""),
    ({"content": [{"type": "text", "text": None}]}, ""),
    ({"content": "Let me check."}, "Let me check."),
    ({"content": [{"type": "text", "text": "Let me check."}]}, "Let me check."),
])
def test_print_trajectory_shows_all_tool_calls_alongside_content(
    capsys, content_fields, expected_text,
):
    trajectory = Trajectory(segments=[Segment(messages=[
        {"role": "assistant", **content_fields, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "search", "arguments": '{"query": "cats"}',
            }},
            {"id": "call_2", "type": "function", "function": {
                "name": "inspect", "arguments": "{}",
            }},
        ]},
    ])])

    print_trajectory(trajectory)
    assert capsys.readouterr().out == (
        f"assistant: {expected_text}\n"
        '  Tool call: search Arguments: {"query": "cats"}\n'
        "  Tool call: inspect Arguments: {}\n"
    )


@pytest.mark.parametrize("tool_calls", [None, []])
def test_print_trajectory_accepts_messages_without_tool_calls(capsys, tool_calls):
    trajectory = Trajectory(segments=[Segment(messages=[
        {"role": "assistant", "content": "Done", "tool_calls": tool_calls},
    ])])

    print_trajectory(trajectory)
    assert capsys.readouterr().out == "assistant: Done\n"


def test_print_trajectory_preserves_segment_boundaries_and_repeated_context(capsys):
    history = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    trajectory = Trajectory(segments=[
        Segment(messages=history),
        Segment(messages=history + [{"role": "user", "content": "Next question"}]),
        Segment(messages=[]),
    ])

    print_trajectory(trajectory)
    assert capsys.readouterr().out == (
        "Segment 0:\nuser: Question\nassistant: Answer\n"
        "Segment 1:\nuser: Question\nassistant: Answer\nuser: Next question\n"
        "Segment 2:\n(empty segment)\n"
    )


@pytest.mark.parametrize("segments,expected", [
    ([], "(no segments)\n"),
    ([Segment(messages=[])], "(empty segment)\n"),
    ([Segment(messages=[{"role": "user", "content": "Context only"}])],
     "user: Context only\n"),
])
def test_print_trajectory_keeps_empty_and_context_only_views(capsys, segments, expected):
    print_trajectory(Trajectory(segments=segments))
    assert capsys.readouterr().out == expected


@pytest.mark.parametrize("rollout", ["chain", "step", None])
def test_print_trajectory_works_with_selected_and_deserialized_results(capsys, rollout):
    result = RunResult(rollout=rollout, trajectories=[
        Trajectory(segments=[Segment(messages=[{"role": "user", "content": text}])])
        for text in ("first", "second")
    ])
    restored = RunResult.model_validate_json(result.model_dump_json())

    print_trajectory(result[1])
    assert capsys.readouterr().out == "user: second\n"
    print_trajectory(restored[0])
    assert capsys.readouterr().out == "user: first\n"
    print_trajectory(result[1])
    assert capsys.readouterr().out == "user: second\n"


def test_print_trajectory_does_not_mutate_data_or_inspect_runtime_steps(capsys):
    runtime_step = object()
    trajectory = Trajectory(
        segments=[Segment(messages=[
            {"role": "assistant", "content": "Answer", "token_ids": [1, 2]},
        ])],
        reward=0.5,
        metrics={"accuracy": 1.0},
        metadata={"task_id": 7},
        steps=[runtime_step],
    )
    before = deepcopy(trajectory.model_dump())
    original_segments = trajectory.segments
    original_messages = trajectory.segments[0].messages
    original_steps = trajectory.steps

    print_trajectory(trajectory)

    assert capsys.readouterr().out == "assistant: Answer\n"
    assert trajectory.model_dump() == before
    assert trajectory.segments is original_segments
    assert trajectory.segments[0].messages is original_messages
    assert trajectory.steps is original_steps
    assert trajectory.steps[0] is runtime_step


def test_agent_does_not_expose_legacy_inspection_methods():
    from agentfly.agents.agent_base import BaseAgent

    assert not hasattr(BaseAgent, "print_messages")
    assert not hasattr(BaseAgent, "get_messages")
