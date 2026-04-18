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


@pytest.mark.asyncio
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
        await ctx.release_resource(scope="rollout")

    print(f"reward result: {result}", flush=True)

    assert isinstance(result, dict)
    assert "reward" in result
    assert result["reward"] in (0.0, 1.0)
