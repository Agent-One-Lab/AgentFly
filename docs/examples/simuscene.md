# SimuScene Setup

SimuScene trains a code-generation agent to produce Python programs that render
short physics-style videos. Reward comes from a **VLM-as-judge ensemble**:
several vision-language models watch the rendered video and answer a list of
visual verification questions. The agent earns the full reward when a majority of
judges pass the video, and small partial credit for extracting code and rendering a video.

## Affected files

- `examples/train_scripts/simuscene/train_simuscene.slurm` — reference training script.
- `examples/train_scripts/simuscene/deploy_vlm_judges.sbatch` — serves the judges and writes
  the env file the training script reads.
- `src/agentfly/rewards/vlm_as_judge/simuscene_rewards.py` — the
  `vlm_as_judge_pass_reward_multi_model` reward used by the script.
- `data/rlhf/simuphy/rl_set.json`, `data/rlhf/simuphy/val_set.json` — train/val
  splits. Each item must contain at least `prompt` and `vlm_questions` fields.

## 1. Stand up the VLM judge endpoints

The reward expects each VLM model to be served by an OpenAI-compatible endpoint
(typically vLLM) on its own host. The reference script uses a 3-model ensemble:

| Index | Model | Notes |
|---|---|---|
| 0 | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Qwen VL flagship |
| 1 | `OpenGVLab/InternVL3_5-241B-A28B` | InternVL ensemble member |
| 2 | `zai-org/GLM-4.6V` | GLM ensemble member |

You can change which models are used — the only constraint is that each entry
in `VLM_MODELS` must have a serving host at the matching index in
`VLM_SERVER_IPS`. A 1-model "ensemble" is also valid.

`deploy_vlm_judges.sbatch` serves the three judges, one model per node, each on port 8000.
When its log prints "ALL JUDGES READY", it has written an env file with `VLM_MODELS` and
`VLM_SERVER_IPS`. You can also serve them your own way before launching training. The reward calls them concurrently per
trajectory, so latency dominates wall-clock time per training step.

## 2. Configure `VLM_MODELS` and `VLM_SERVER_IPS`

Both env vars are comma-separated lists, position-aligned (model `i` is served
at host `i`):

```bash
export VLM_MODELS="Qwen/Qwen3-VL-235B-A22B-Instruct,OpenGVLab/InternVL3_5-241B-A28B,zai-org/GLM-4.6V"
export VLM_SERVER_IPS="vlm-host-1,vlm-host-2,vlm-host-3"
```

The training script reads them from that env file: set `JUDGES_ENV` to its path when
submitting (`JUDGES_ENV=/path/to/judges.env sbatch ...`). The script checks each judge's
`/health` endpoint and stops if one is down. If the two lists differ in length or either is
empty, the reward logs an error and returns 0.

The judges read the rendered videos through `file://` paths, so the script puts them under
`VLM_SHARED_VIDEO_DIR` on the shared filesystem.

`VLM_MODELS` and `VLM_SERVER_IPS` are read at import time. Set them in the
shell that launches `python -m agentfly.cli train`, not inside the Python
process.

## 3. Reward composition

`vlm_as_judge_pass_reward_multi_model` renders the code in the final response to a video and
asks every judge the task's questions. Each judge passes or fails the video, and:

| Outcome | `reward` |
|---|---|
| A majority of judges pass the video | `1.0` |
| No majority, video opens | `0.06` |
| No majority, video saved but doesn't open | `0.04` |
| No code, no video, or any judge returns no parseable answer | `0.0` |

The partial credits are `MULTI_MODEL_LADDER_*` near the top of `simuscene_rewards.py`; edit them
in source for a different shaping. Other SimuScene rewards (all-judges pass, the fraction of
questions passed, and a text-only judge ablation) are listed in
[Predefined Rewards](../api_references/rewards/predefined_rewards.md); pick one with
`REWARD_NAME=...`.

## 4. Launch training

```bash
JUDGES_ENV=/path/to/judges.env sbatch examples/train_scripts/simuscene/train_simuscene.slurm
```

The reference script defaults to:

- 2 nodes × 8 GPUs (configure via `#SBATCH --nodes`).
- GRPO with `num_chains=8`, `train_batch_size=32`, `mini_batch_size=128`.
- `max_turns=1` (single-shot code generation; the agent answers once via the
  `answer_qa` tool and the rendered video is judged).
- WandB project `SimuScene`.

Update `train_file`, `val_file` and the SLURM resource lines for your environment. The default
model is a local SFT checkpoint: set `MODEL=` (and `TEMPLATE=` if needed) to a Hugging Face model
id or your own checkpoint.

## 5. Smoke test the judge connection

A fast way to confirm the env vars and endpoints are wired up is to import the
reward module and call it once with a known-good response:

```python
import os
os.environ["VLM_MODELS"] = "Qwen/Qwen3-VL-235B-A22B-Instruct"
os.environ["VLM_SERVER_IPS"] = "vlm-host-1"

from agentfly.rewards.vlm_as_judge.simuscene_rewards import vlm_specs_from_globals
specs = vlm_specs_from_globals()
assert specs, "VLM specs are empty — check VLM_MODELS / VLM_SERVER_IPS"
print(specs)
```

If `specs` is empty or the resulting HTTP request fails, the training run will
reward 0 across the board, so resolve this before launching a long job.
