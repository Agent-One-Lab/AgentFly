"""Smoke-test the deployed SimuScene VLM judges through the real training reward.

    source logs/simuscene-judges-<jobid>/judges.env
    uv run --no-sync python examples/train_scripts/simuscene/smoke_test_judges.py

Runs vlm_as_judge_pass_reward_multi_model on a small rendered video twice: once with a
single easy question (all judges should pass) and once with a real 4-12 question set from
val_set.json (judges should fail it, but must still answer in parseable JSON). Exits 1 if
any judge fails to respond or to produce parseable output.
"""

import asyncio
import json
import os
import sys

os.environ.setdefault("VLM_SHARED_VIDEO_DIR", os.path.abspath("./tmp/vlm_judge/smoke"))

from agentfly.core import Context  # noqa: E402
from agentfly.rewards.vlm_as_judge import simuscene_reward as sr  # noqa: E402

FINAL_RESPONSE = """<think>draw a square</think>
```python
import cv2
import numpy as np
import sys

output_path = sys.argv[1]
width, height = 320, 240
fps = 12
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
for i in range(fps * 5):
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    x = int(20 + i / (fps * 5 - 1) * 240)
    cv2.rectangle(frame, (x, 100), (x + 40, 140), (0, 0, 255), -1)
    writer.write(frame)
writer.release()
```"""

EASY_QUESTIONS = {
    "summarize": "A red square moves from left to right on a white background.",
    "vlm_questions": [
        {"index": "1", "question": "A red square is visible in the video."},
        {"index": "2", "question": "The red square moves from the left side toward the right side."},
    ],
}


async def run_case(name, vlm_questions):
    ctx = Context(rollout_id=f"smoke-{name}")
    try:
        result = await sr.vlm_as_judge_pass_reward_multi_model(
            final_response=FINAL_RESPONSE, vlm_questions=vlm_questions, context=ctx
        )
    finally:
        await ctx.end_resource(scope="rollout")
    print(f"[{name}] {json.dumps(result, indent=1)}", flush=True)
    ok = result.get("video_saved") == 1.0
    for model in sr.VLM_MODELS:
        if result.get(f"model_format_success@{model}") != 1.0:
            print(f"[{name}] FAIL: {model} did not return parseable output", flush=True)
            ok = False
    return ok, result


async def main():
    print("VLM_MODELS", sr.VLM_MODELS, "VLM_SERVER_IPS", sr.VLM_SERVER_IPS, flush=True)
    if not sr.VLM_MODELS or len(sr.VLM_MODELS) != len(sr.VLM_SERVER_IPS):
        print("FAIL: VLM_MODELS / VLM_SERVER_IPS not set"); return 1
    val = json.load(open("./data/rlhf/simuphy/val_set.json"))
    ok1, r1 = await run_case("easy", EASY_QUESTIONS)
    ok2, _ = await run_case("real-questions", val[0]["vlm_questions"])
    if ok1 and r1.get("reward") != 1.0:
        print("WARN: easy case did not reach a majority pass; check judge answers above", flush=True)
    print("SMOKE TEST", "PASSED" if (ok1 and ok2) else "FAILED", flush=True)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
