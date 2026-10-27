# SWE Task Setup

The SWE-bench / R2E-Gym training and evaluation paths in AgentFly run patched
repositories inside enroot containers and grade them with `swebench`'s harness.
Because these dependencies are heavy and only required for SWE work, they are
**not** part of the default install. This page covers what to install, how to
pre-pull the container images, and how to point AgentFly at them.

## Affected scripts and tests

These are the entry points that need this setup:

- `examples/eval_scripts/swebench/run_swebench_verified.sh` — SWE-bench Verified evaluation.
- `examples/train_scripts/swe/train_swe_r2e_gym.slurm`
- `examples/train_scripts/swe/train_swe_r2e_gym_tools.slurm`
- `examples/train_scripts/swe/train_swe_r2e_gym_tools_32B.slurm`
- `examples/train_scripts/context/context_run_swe.sh`
- `tests/unit/rewards/swe/test_eval_r2e_gym.py`
- `tests/unit/rewards/swe/test_get_patch_from_runtime.py`

## 1. Extra Python dependencies

Install these into the same `.venv` used for AgentFly. R2E-Gym pins older versions of
packages AgentFly uses (for example `openai`, `anthropic`, `huggingface-hub`), and a plain
install would downgrade them. Keep everything already installed at its current version by
passing the installed list as overrides:

```bash
uv pip freeze --exclude-editable > /tmp/keep.txt
uv pip install swebench "git+https://github.com/R2E-Gym/R2E-Gym.git" --override /tmp/keep.txt
```

This only adds packages. (`uv sync` removes them again, since they aren't AgentFly
dependencies; rerun the two commands after a sync.)

`swebench` provides the grading harness (`swebench.harness.*`) and `r2egym`
provides the trajectory utilities and log parsers used by the R2E-Gym reward.
Both are imported lazily, so the rest of AgentFly works without them, but any
SWE reward call will fail with `ImportError` until they are installed.

## 2. enroot

The SWE rewards launch sandboxed shells inside enroot containers. Install the
enroot system package as described in
[Installation](../start/installation.md#requirements) — sudo is required.

## 3. Pre-pull the container images

Every task names its Docker image in the dataset's `image_id` field (for example
`namanjain12/aiohttp_final:f0d7488…` in R2E-Gym), and each image is converted into
an enroot `.sqsh` file. Pull them **before training**: a missing image is pulled the
first time a rollout needs it, which stalls that rollout for minutes and floods the
network when many rollouts start at once.

Keep one directory per image set on a disk with enough space:

| Task | Task file (size) | Tasks | Images | Image registry | Image disk |
|---|---|---|---|---|---|
| R2E-Gym training (lite) | `data/rlhf/os/r2e-gym-lite.json` (2.3 GB) | 2,700 | 2,700 | Docker Hub | ~4.3 TB |
| SWE-bench Verified evaluation | `data/rlhf/os/swe-bench-verified.json` (9 MB) | 500 | 500 | `ghcr.io` | ~1.2 TB |
| SWE-smith training | `data/rlhf/os/swe-smith-py.json` (4.1 GB) | 39,471 | 131 | Docker Hub | ~450 GB |

The task files themselves are on the
[`Agent-One/AgentFly-Train`](https://huggingface.co/datasets/Agent-One/AgentFly-Train)
dataset. `agentfly train` and `agentfly pull-images` download a missing file from
there automatically, after one confirmation prompt (`AF_ASSUME_YES=1` accepts it
non-interactively). They are converted from the upstream datasets
(`R2E-Gym/R2E-Gym-Lite`, `princeton-nlp/SWE-bench_Verified`, `SWE-bench/SWE-smith-py`):
each task gets an `image_id` and a `messages` prompt (the problem statement). SWE-smith tasks
also get `git_commit_hash`: their images start on `main`, and each task's bug lives on a git
branch named after the task, which the SWE tools check out before the agent's first command.
`r2e-gym-lite.json` is a 2,700-task subset of R2E-Gym-Lite's 4,578 training tasks.

??? note "SWE-smith for MinisweAgent"
    `MinisweAgent` needs its own system prompt and instance message (mini-swe-agent's,
    byte-for-byte as in its SFT trajectories) instead of the bare problem statement.
    Build them with the agent's helper so the dataset and the agent can't drift apart:

    ```python
    import json
    from agentfly.agents.specialized.miniswe import build_messages

    tasks = json.load(open("data/rlhf/os/swe-smith-py.json"))
    for t in tasks:
        t["messages"] = build_messages(t["problem_statement"])
    json.dump(tasks, open("data/rlhf/os/swe-smith-py-miniswe.json", "w"))
    ```

Then pull the images with [`agentfly pull-images`](../start/cli.md#agentfly-pull-images).
It reads the task file (downloading it first if missing, which for the large files above
takes a while even with `--dry-run`):

```bash
export ENROOT_IMAGES_PATH=/some/big/disk/enroot/images/r2e-gym-lite
agentfly pull-images data/rlhf/os/r2e-gym-lite.json --dry-run   # how many are missing
agentfly pull-images data/rlhf/os/r2e-gym-lite.json -j 4

# SWE-bench Verified images are on GitHub's registry, not Docker Hub:
export ENROOT_IMAGES_PATH=/some/big/disk/enroot/images/swe-bench-verified
agentfly pull-images data/rlhf/os/swe-bench-verified.json --registry ghcr.io -j 4
```

Images already present are skipped, so rerun the same command to resume after an
interruption or to retry failures (they are listed in `failed_pulls.txt`). Files are
named the way enroot looks them up at run time, `repo+name+tag.sqsh`:

```
/some/big/disk/enroot/images/
├── r2e-gym-lite/
│   ├── namanjain12+aiohttp_final+f0d74880deec8fcd982bce639c93c5e130d41198.sqsh
│   └── ...
└── swe-bench-verified/
    ├── epoch-research+swe-bench.eval.x86_64.astropy__astropy-12907.sqsh
    └── ...
```

The R2E-Gym and SWE-smith images are on Docker Hub, which rate-limits anonymous pulls.
For a large set, add Docker Hub credentials to enroot's credentials file
(`~/.config/enroot/.credentials`, one line:
`machine auth.docker.io login <user> password <access-token>`) and keep `-j` small. The
SWE-bench Verified images on `ghcr.io` are public and need no credentials. If an image is
missing at run time, enroot pulls it from Docker Hub on first use: slow for R2E-Gym and
SWE-smith, and a failure for SWE-bench Verified, whose images are only on `ghcr.io`.

## 4. Set `ENROOT_IMAGES_PATH`

The example scripts and tests read `ENROOT_IMAGES_PATH` from the environment
and fall back to a placeholder otherwise. Point it at the directory you pulled
into, and export it in your shell before launching (enroot reads it once, when
AgentFly starts), e.g.:

```bash
export ENROOT_IMAGES_PATH=/some/big/disk/enroot/images/r2e-gym-lite
sbatch examples/train_scripts/swe/train_swe_r2e_gym.slurm
```

```bash
export ENROOT_IMAGES_PATH=/some/big/disk/enroot/images/swe-bench-verified
bash examples/eval_scripts/swebench/run_swebench_verified.sh
```

For the CLI evaluator you can also pass it inline:

```bash
python -m agentfly.cli swebench \
    --enroot-images-path /some/big/disk/enroot/images/swe-bench-verified \
    ...
```

## 5. Run one task

Before a full evaluation or training run, check the whole chain (model, container, tools,
grading) on a single task with `agentfly swebench --limit 1`. This runs Qwen3-8B on the
first SWE-bench Verified task with a local vLLM engine on one GPU.

First get that task's image. `--limit 1` pulls only the image of the task `agentfly swebench
--limit 1` will run (about 2 GB instead of the full set), and also downloads the task file if it
is missing; `agentfly swebench` itself doesn't download it:

```bash
agentfly pull-images data/rlhf/os/swe-bench-verified.json --registry ghcr.io --limit 1 \
    --to /some/big/disk/enroot/images/swe-bench-verified
```

Then run the task:

```bash
agentfly swebench \
    --data-path data/rlhf/os/swe-bench-verified.json \
    --result-dir results/swe/one-task \
    --model-name-or-path Qwen/Qwen3-8B \
    --template qwen3 \
    --backend async_vllm --max-model-len 32768 \
    --agent bash --tool-set bash \
    --max-turns 30 --limit 1 \
    --enroot-images-path /some/big/disk/enroot/images/swe-bench-verified
```

With `--backend async_vllm`, pass `--template`: without it the model's own Hugging Face chat
template is used, and Qwen3's rewrites earlier turns (it drops their thinking), which AgentFly
rejects with *"cannot isolate turn ..."*. With `--backend client` the server renders the prompts
and no template is needed.

`results/swe/one-task/` then holds `run_summary.json` and one JSON per task with the reward and
the full trajectory (`trajectory.segments[-1].messages`, `finish_reason`, `tool_call_counts`).
A reward of 0.0 with `finish_reason: max_turns` just means the model didn't solve the task;
an exception means the setup is broken. `--seed N` picks a random task instead of the first; pass the same `--seed` to `pull-images`.

## 6. Smoke test

With the environment variable set and images present:

```bash
pytest tests/unit/rewards/swe/test_eval_r2e_gym.py -x
```

A successful run confirms that `r2egym`, `swebench`, enroot, and your image
directory are all wired up correctly.
