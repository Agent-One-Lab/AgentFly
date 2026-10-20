from __future__ import annotations

import pytest

from agentfly.core import Context
from agentfly.rewards.vlm_as_judge import simuscene_reward as sr

# Fill these values manually before running this integration test.
VLM_MODELS = ["Qwen/Qwen2.5-VL-72B-Instruct"]
VLM_SERVER_IPS = ["127.0.0.1"]
API_MAX_GLOBAL_NUM = 1


def _build_test_final_response() -> str:
    return """```python
import cv2
import numpy as np
import sys

output_path = sys.argv[1]
width, height = 256, 256
fps = 12
writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)
assert writer.isOpened(), "VideoWriter failed to open"

num_frames = fps * 4
for i in range(num_frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    t = i / max(1, num_frames - 1)
    x = int(20 + t * 160)
    y = int(110 + 30 * np.sin(t * 2 * np.pi))
    cv2.rectangle(frame, (x, y), (x + 40, y + 40), (0, 0, 255), -1)
    writer.write(frame)

writer.release()
```"""


def _build_test_questions() -> dict:
    return {
        "summarize": "A single red square moves from left to right on a black background.",
        "vlm_questions": [
            {
                "index": "1",
                "question": "A red square moves from left to right.",
                "weight": 1.0,
            }
        ],
    }


@pytest.mark.gpu
@pytest.mark.asyncio(scope="session")
async def test_vlm_as_judge_pass_reward_real_vlm():
    if not VLM_MODELS:
        pytest.skip("Set VLM_MODELS in this test file to run the integration test.")

    # Configure VLM endpoints via module globals (paired by index).
    sr.VLM_MODELS = VLM_MODELS
    sr.VLM_SERVER_IPS = VLM_SERVER_IPS

    # Optionally cap concurrent API resources for this reward path.
    original_builder = sr._build_api_model_spec

    def _patched_build_api_model_spec(*args, **kwargs):
        spec = original_builder(*args, **kwargs)
        spec.max_global_num = API_MAX_GLOBAL_NUM
        return spec

    sr._build_api_model_spec = _patched_build_api_model_spec

    final_response = _build_test_final_response()
    vlm_questions = _build_test_questions()
    ctx = Context(rollout_id="test_vlm_as_judge_reward_real_vlm")

    try:
        result = await sr.vlm_as_judge_pass_reward(
            final_response=final_response,
            vlm_questions=vlm_questions,
            context=ctx,
        )
    finally:
        sr._build_api_model_spec = original_builder
        await ctx.end_resource(scope="rollout")

    print(f"reward result: {result}", flush=True)

    assert isinstance(result, dict)
    assert "reward" in result
    assert result["reward"] in (0.0, 1.0)


def _judge(model: str, answers: dict) -> dict:
    return {
        "model": model,
        "error": None,
        "response_success": True,
        "format_success": True,
        "passed": all(answers.get(str(i)) is True for i in range(1, 4)),
        "result_map": answers,
    }


_THREE_QUESTIONS = [{"index": str(i), "question": f"q{i}"} for i in range(1, 4)]


def test_per_question_vote_majority_per_question():
    # Each judge misses a different question: every question still has 2/3 True votes.
    results = [
        _judge("a", {"1": False, "2": True, "3": True}),
        _judge("b", {"1": True, "2": False, "3": True}),
        _judge("c", {"1": True, "2": True, "3": True}),
    ]
    assert sr._per_question_vote(results, _THREE_QUESTIONS) == [True, True, True]


def test_per_question_vote_missing_answer_counts_false():
    results = [
        _judge("a", {"1": True, "2": True}),
        _judge("b", {"1": True}),
        _judge("c", {"1": False, "2": True, "3": True}),
    ]
    assert sr._per_question_vote(results, _THREE_QUESTIONS) == [True, True, False]


@pytest.mark.asyncio
async def test_ratio_reward_uses_per_question_vote(monkeypatch):
    specs = [sr._build_api_model_spec(model=m, server_ip="127.0.0.1") for m in ("a", "b", "c")]
    results = [
        _judge("a", {"1": False, "2": True, "3": True}),
        _judge("b", {"1": True, "2": False, "3": False}),
        _judge("c", {"1": True, "2": True, "3": True}),
    ]

    async def fake_backbone(context, **kwargs):
        return {
            "ok": True,
            "prepared": {"questions_list": _THREE_QUESTIONS},
            "video_path": None,
            "model_results": results,
            "available_results": results,
            "video_metrics": sr._video_generation_metrics(True, True, True),
        }

    monkeypatch.setattr(sr, "vlm_specs_from_globals", lambda: specs)
    monkeypatch.setattr(sr, "_run_vlm_backbone", fake_backbone)
    out = await sr.vlm_as_judge_ratio_reward_multi_model(
        final_response="", vlm_questions={"vlm_questions": _THREE_QUESTIONS}, context=Context(rollout_id="t")
    )
    # q1: 2/3 True, q2: 2/3 True, q3: 2/3 True -> ratio 1.0, although only judge c passes the video.
    assert out["reward"] == 1.0
    assert out["vote_all_true"] == 1.0
    assert out["pass#2"] == 0.0
    success_keys = set(out)

    # One judge failing to answer zeroes the reward, as in the binary reward.
    failed = [results[0], results[1], {"model": "c", "error": "parse_error"}]

    async def fake_backbone_failed(context, **kwargs):
        out = await fake_backbone(context, **kwargs)
        out["model_results"] = failed
        out["available_results"] = failed[:2]
        return out

    monkeypatch.setattr(sr, "_run_vlm_backbone", fake_backbone_failed)
    out = await sr.vlm_as_judge_ratio_reward_multi_model(
        final_response="", vlm_questions={"vlm_questions": _THREE_QUESTIONS}, context=Context(rollout_id="t2")
    )
    assert out["reward"] == 0.0
    # Every rollout must report the same metric keys, success or failure.
    assert set(out) == success_keys


@pytest.mark.asyncio
async def test_ratio_reward_zero_when_render_fails(monkeypatch):
    specs = [sr._build_api_model_spec(model=m, server_ip="127.0.0.1") for m in ("a", "b", "c")]

    async def fake_backbone(context, **kwargs):
        return {
            "ok": False,
            "reward": 0.0,
            "prepared": None,
            "model_results": [],
            "available_results": [],
            "video_metrics": sr._video_generation_metrics(code_extracted=True),
        }

    monkeypatch.setattr(sr, "vlm_specs_from_globals", lambda: specs)
    monkeypatch.setattr(sr, "_run_vlm_backbone", fake_backbone)
    out = await sr.vlm_as_judge_ratio_reward_multi_model(
        final_response="", vlm_questions={"vlm_questions": _THREE_QUESTIONS}, context=Context(rollout_id="t3")
    )
    assert out["reward"] == 0.0
    assert out["code_extracted"] == 1.0
    assert out["vote_all_true"] == 0.0


_CODE_RESPONSE = "reasoning</think>\n```python\nimport cv2\nprint('{not a format field}')\n```"


def _llm_judge(model, answers):
    # answers over the extended set: 1 = exec, 2 = video, 3..5 = the three domain questions
    out = _judge(model, answers)
    out["passed"] = all(answers.get(str(i)) is True for i in range(1, 6))
    return out


def test_llm_judge_questions_prepend_exec_and_video():
    qs = sr._llm_judge_questions(_THREE_QUESTIONS)
    assert [q["index"] for q in qs] == ["1", "2", "3", "4", "5"]
    assert qs[0]["question"] == sr.LLM_JUDGE_EXEC_QUESTION
    assert qs[1]["question"] == sr.LLM_JUDGE_VIDEO_QUESTION
    assert [q["question"] for q in qs[2:]] == ["q1", "q2", "q3"]


@pytest.mark.asyncio
async def test_llm_judge_reward_binary_majority(monkeypatch):
    specs = [sr._build_api_model_spec(model=m, server_ip="127.0.0.1") for m in ("a", "b", "c")]
    all_true = {str(i): True for i in range(1, 6)}
    answers = {"a": all_true, "b": all_true, "c": {**all_true, "2": False}}
    prompts = []

    async def fake_judge(context, *, messages, spec, questions_list, timeout_seconds=600):
        prompts.append(messages[0]["content"][0]["text"])
        assert len(questions_list) == 5
        return _llm_judge(spec.model_name_or_path, answers[spec.model_name_or_path])

    monkeypatch.setattr(sr, "vlm_specs_from_globals", lambda: specs)
    monkeypatch.setattr(sr, "_run_one_vlm_model", fake_judge)
    out = await sr.llm_as_judge_pass_reward_multi_model(
        final_response=_CODE_RESPONSE, vlm_questions={"vlm_questions": _THREE_QUESTIONS}, context=Context(rollout_id="l1")
    )
    # 2 of 3 judges pass every question -> reward 1
    assert out["reward"] == 1.0 and out["pass#2"] == 1.0 and out["pass#3"] == 0.0
    assert out["judged_exec"] == 1.0 and out["judged_video"] == 1.0 and out["vote_all_true"] == 1.0
    assert "video_saved" not in out and "video_opened" not in out
    # the code (with its braces) and the extended questions are in the prompt, nothing executed
    assert "print('{not a format field}')" in prompts[0]
    assert "1. " + sr.LLM_JUDGE_EXEC_QUESTION in prompts[0] and "5. q3" in prompts[0]
    success_keys = set(out)

    # only 1 of 3 judges passes -> ladder credit for extracted code, as in the VLM binary reward
    answers["b"] = {**all_true, "4": False}
    out = await sr.llm_as_judge_pass_reward_multi_model(
        final_response=_CODE_RESPONSE, vlm_questions={"vlm_questions": _THREE_QUESTIONS}, context=Context(rollout_id="l2")
    )
    assert out["reward"] == sr.MULTI_MODEL_LADDER_CODE_REWARD
    assert set(out) == success_keys

    # a judge error zeroes the reward
    async def failing_judge(context, *, messages, spec, questions_list, timeout_seconds=600):
        if spec.model_name_or_path == "c":
            return {"model": "c", "error": "parse_error"}
        return _llm_judge(spec.model_name_or_path, all_true)

    monkeypatch.setattr(sr, "_run_one_vlm_model", failing_judge)
    out = await sr.llm_as_judge_pass_reward_multi_model(
        final_response=_CODE_RESPONSE, vlm_questions={"vlm_questions": _THREE_QUESTIONS}, context=Context(rollout_id="l3")
    )
    assert out["reward"] == 0.0 and out["code_extracted"] == 1.0
    assert set(out) == success_keys

    # no code -> 0, judges not called
    monkeypatch.setattr(sr, "_run_one_vlm_model", None)
    out = await sr.llm_as_judge_pass_reward_multi_model(
        final_response="no code here", vlm_questions={"vlm_questions": _THREE_QUESTIONS}, context=Context(rollout_id="l4")
    )
    assert out["reward"] == 0.0 and out["code_extracted"] == 0.0
    assert set(out) == success_keys
