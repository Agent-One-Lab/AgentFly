# SimuScene Setup

SimuScene trains a code-generation agent to produce Python programs that render
short physics-style videos. Reward comes from a **VLM-as-judge ensemble**:
several vision-language models watch the rendered video and answer a list of
visual verification questions. The agent's reward is shaped by code execution,
video rendering, video saving, and the ensemble's pass rate on the questions.

## Affected files

- `examples/train_scripts/simuscene/test_vlm_as_judge_train.sh` — reference training script.
- `src/agentfly/rewards/vlm_as_judge/simuscene_reward.py` — the
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

Bring up each endpoint independently (one host per VLM, or however your cluster
serves them) before launching training. The reward calls them concurrently per
trajectory, so latency dominates wall-clock time per training step.

## 2. Configure `VLM_MODELS` and `VLM_SERVER_IPS`

Both env vars are comma-separated lists, position-aligned (model `i` is served
at host `i`):

```bash
export VLM_MODELS="Qwen/Qwen3-VL-235B-A22B-Instruct,OpenGVLab/InternVL3_5-241B-A28B,zai-org/GLM-4.6V"
export VLM_SERVER_IPS="vlm-host-1,vlm-host-2,vlm-host-3"
```

The training script ships with these `export` lines already filled in for the
default 3-model setup; replace the `<vlm-server-N>` placeholders with the
actual hosts before submitting. If lengths differ or either list is empty, the
reward logs an error and returns a 0 reward, so misconfiguration fails loudly
rather than silently.

`VLM_MODELS` and `VLM_SERVER_IPS` are read at import time. Set them in the
shell that launches `python -m agentfly.cli train`, not inside the Python
process.

## 3. Reward composition

`vlm_as_judge_pass_reward_multi_model` returns a `reward` field plus diagnostic
metrics. The reward is the sum of four components, evaluated in order — each
gate must pass before the next contributes:

| Component | Default weight | Triggered when |
|---|---|---|
| Code extraction | `0.04` | A Python code block is parsed from the final response |
| Code render | `0.06` | The extracted code executes without raising |
| Video saved & openable | `0.10` | A video file is produced and `cv2` can open it |
| VLM ensemble pass rate | `0.80` | At least `VLM_ENSEMBLE_MIN_RESPONSES=2` models return parseable JSON; reward scales with the per-question true rate, averaged across models |

The constants are defined near the top of `simuscene_reward.py`
(`LADDER_*_REWARD`, `VLM_ENSEMBLE_MIN_RESPONSES`) — edit them in source if you
need a different shaping.

## 4. Launch training

```bash
sbatch examples/train_scripts/simuscene/test_vlm_as_judge_train.sh
```

The reference script defaults to:

- 2 nodes × 8 GPUs (configure via `#SBATCH --nodes`).
- GRPO with `num_chains=8`, `train_batch_size=32`, `mini_batch_size=128`.
- `max_turns=1` (single-shot code generation; the agent answers once via the
  `answer_qa` tool and the rendered video is judged).
- WandB project `SimuScene`.

Update `model`, `train_file`, `val_file`, and the SLURM resource lines for
your environment. Note that the default `model` path is a local SFT checkpoint
— swap it for any HuggingFace model id or your own checkpoint.

## 5. Smoke test the judge connection

A fast way to confirm the env vars and endpoints are wired up is to import the
reward module and call it once with a known-good response:

```python
import os
os.environ["VLM_MODELS"] = "Qwen/Qwen3-VL-235B-A22B-Instruct"
os.environ["VLM_SERVER_IPS"] = "vlm-host-1"

from agentfly.rewards.vlm_as_judge.simuscene_reward import _vlm_specs_from_globals
specs = _vlm_specs_from_globals()
assert specs, "VLM specs are empty — check VLM_MODELS / VLM_SERVER_IPS"
print(specs)
```

If `specs` is empty or the resulting HTTP request fails, the training run will
reward 0 across the board, so resolve this before launching a long job.
