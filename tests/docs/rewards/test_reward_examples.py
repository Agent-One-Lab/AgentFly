"""Runnable examples backing docs/api_references/rewards/*.md.

Each test wraps the example shown in the corresponding doc page with section
markers so pymdownx.snippets can include the code directly. Tests assert
output shape (dict / float) rather than exact numeric values, so the snippets
stay callable across implementation tweaks while still failing if the function
is renamed or re-signatured.
"""
import pytest

from agentfly.rewards import (
    qa_f1_reward,
    qa_f1_reward_tool,
    math_equal_reward,
    math_equal_reward_tool,
)


def test_qa_f1_reward_basic():
    # --8<-- [start:qa_f1_reward_basic]
    result = qa_f1_reward(
        final_response="Paris is the capital",
        answer="Paris",
        trajectory=[],
    )
    print(result)
    # {"reward": <f1>, "f1": <f1>, "em": <em>, "precision": ..., "recall": ...}
    # --8<-- [end:qa_f1_reward_basic]
    assert isinstance(result, dict)
    assert "reward" in result and 0.0 <= result["reward"] <= 1.0
    assert {"f1", "em", "precision", "recall"} <= set(result)


def test_qa_f1_reward_tool_with_trajectory():
    # --8<-- [start:qa_f1_reward_tool_with_trajectory]
    trajectory = [
        {"role": "assistant", "content": "I need to search for information"},
        {"role": "tool", "content": "search results"},
        {"role": "assistant", "content": "Based on my search, the answer is Paris"},
    ]
    result = qa_f1_reward_tool(
        final_response="Paris",
        answer="Paris",
        trajectory=trajectory,
    )
    print(result)
    # --8<-- [end:qa_f1_reward_tool_with_trajectory]
    assert isinstance(result, dict)
    assert "reward" in result and 0.0 <= result["reward"] <= 1.0


def test_math_equal_reward_basic():
    # --8<-- [start:math_equal_reward_basic]
    result = math_equal_reward(
        final_response="\\boxed{42}",
        answer="\\boxed{42}",
    )
    print(result)  # {"reward": 1.0}
    # --8<-- [end:math_equal_reward_basic]
    assert isinstance(result, dict) and result.get("reward") == 1.0


def test_math_equal_reward_tool_with_trajectory():
    # --8<-- [start:math_equal_reward_tool_with_trajectory]
    trajectory = [
        {"role": "assistant", "content": "I need to calculate..."},
        {"role": "tool", "content": "calculation result"},
        {"role": "assistant", "content": "The answer is \\boxed{42}"},
    ]
    result = math_equal_reward_tool(
        final_response="\\boxed{42}",
        answer="\\boxed{42}",
        trajectory=trajectory,
    )
    print(result)
    # --8<-- [end:math_equal_reward_tool_with_trajectory]
    assert isinstance(result, (float, dict))
