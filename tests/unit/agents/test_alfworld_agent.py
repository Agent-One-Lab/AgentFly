"""End-to-end ALFWorld agent rollout, WITHOUT the verl trainer.

This exercises exactly the rollout path the trainer drives — ActionAgent +
alfworld_step tool + ALFWorld env + alfworld_episode_reward, on a real async
vLLM backend — but from a plain `agent.run(...)` call. It is an isolation probe:
if training hangs at validation but this test completes, the hang is in the
trainer/validation glue, not in the agent/tool/env/reward code. If this test
also hangs, the problem is in our rollout code.

Parameters mirror examples/train_scripts/alfworld/train_alfworld_ray.sh. Defaults
are scaled down for a single test; override via env vars to reproduce the
validation-scale run:
    AF_TEST_ALFWORLD_NUM_TASKS   (default 4;   validation uses 134)
    AF_TEST_ALFWORLD_NUM_CHAINS  (default 1;   validation is greedy, 1 chain)
    AF_TEST_ALFWORLD_MAX_TURNS   (default 50;  matches the train script)

Requires: a GPU (vLLM), and the ALFWorld enroot image + the preprocessed
`data/rlhf/alfworld/alfworld_valid_unseen_tasks.json` dataset (regenerate with
`python examples/data_preprocess/prepare_alfworld.py --out-dir data/rlhf/alfworld`).
"""

import json
import os

import pytest

from agentfly.agents.specialized.action_agent import ActionAgent
from agentfly.tools import alfworld_step
from agentfly.rewards import alfworld_episode_reward


# --- config mirrored from train_alfworld_ray.sh -----------------------------
MODEL = "Qwen/Qwen2.5-3B-Instruct"
TEMPLATE = "action-agent"
MAX_MODEL_LEN = 8192
MAX_NEW_TOKENS_PER_TURN = 256
GPU_MEMORY_UTILIZATION = 0.75

# Scaled-down defaults; override via env to reproduce validation scale.
NUM_TASKS = int(os.getenv("AF_TEST_ALFWORLD_NUM_TASKS", "4"))
NUM_CHAINS = int(os.getenv("AF_TEST_ALFWORLD_NUM_CHAINS", "1"))
MAX_TURNS = int(os.getenv("AF_TEST_ALFWORLD_MAX_TURNS", "50"))

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
EVAL_DATASET = os.path.join(
    _REPO_ROOT, "data", "rlhf", "alfworld", "alfworld_valid_unseen_tasks.json"
)

SYSTEM_PROMPT = (
    "You are an ALFWorld agent operating in an interactive, text-based household "
    "environment (derived from the ALFRED benchmark). You are placed in one of 120 "
    "possible rooms (kitchen, bedroom, bathroom, or living room) populated with "
    "portable objects (e.g., apple, mug, book) and static receptacles (e.g., "
    "microwave, fridge, drawer, countertop). Your goal is to complete the given "
    "household task by interacting with the world through high-level text commands, "
    "and to finish it in as few steps as possible. The environment is partially "
    "observable: the initial observation lists all navigable receptacles in the room, "
    "but you must actively go to receptacles, open them, and examine their contents to "
    "find target objects.\n\n"
    "You must conduct reasoning inside <think> and </think> first every time you get "
    "new information. After reasoning, you can do one action by <action> action "
    "</action>. If you think you have finished the task, summarize what you have done.\n\n"
    "Each observation lists the valid actions for the current state under 'Admissible "
    "actions:'. Always choose your next action from that list — any command not on it "
    "will fail with 'Nothing happens'. Objects and receptacles are referred to by class "
    "name plus an ID (e.g., mug 1, countertop 2).\n\n"
    "Remember that you must put your action inside <action> and </action> tags."
)


def _load_tasks(n: int):
    with open(EVAL_DATASET, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    out = []
    for t in tasks[:n]:
        out.append(
            {
                "messages": t["messages"],
                "question": t["question"],
                "task_id": t["task_id"],
                "split": t.get("split", "valid_unseen"),
            }
        )
    return out


@pytest.mark.gpu
@pytest.mark.skipif(
    not os.path.exists(EVAL_DATASET),
    reason=f"ALFWorld eval dataset not found at {EVAL_DATASET}",
)
@pytest.mark.asyncio(loop_scope="session")
async def test_alfworld_agent_rollout():
    """Run the ALFWorld agent over a few tasks and assert the rollout completes."""
    messages = _load_tasks(NUM_TASKS)
    assert messages, "no ALFWorld tasks loaded"

    agent = ActionAgent(
        MODEL,
        tools=[alfworld_step],
        reward_fn=alfworld_episode_reward,
        template=TEMPLATE,
        system_prompt=SYSTEM_PROMPT,
        max_model_len=MAX_MODEL_LEN,
        backend_config={
            "backend": "async_vllm",
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        },
        monitors=[],  # no wandb in tests
    )

    result = await agent.run(
        messages=messages,
        max_turns=MAX_TURNS,
        num_chains=NUM_CHAINS,
        generation_config={"max_tokens": MAX_NEW_TOKENS_PER_TURN},
    )

    # PRIMARY signal: run() returned at all (i.e. the rollout did not hang).
    assert result is not None
    trajectories = result.trajectories
    print(f"ran {len(trajectories)} trajectories over {len(messages)} tasks")
    assert trajectories, "run() returned no trajectories"

    # Secondary (best-effort, schema-tolerant): surface the rewards for the user
    # to eyeball. Kept lenient so an unexpected trajectory shape reports rather
    # than masking the pass/fail = completed/hung signal.
    rewards = [t.get("reward") for t in trajectories if isinstance(t, dict)]
    print(f"trajectory count: {len(trajectories)} | rewards: {rewards}")
