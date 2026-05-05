# SWE Task Setup

The SWE-bench / R2E-Gym training and evaluation paths in AgentFly run patched
repositories inside enroot containers and grade them with `swebench`'s harness.
Because these dependencies are heavy and only required for SWE work, they are
**not** part of the default install. This page covers what to install and how
to point AgentFly at your container images.

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

Install these into the same conda environment used for AgentFly:

```bash
pip install swebench
pip install git+https://github.com/R2E-Gym/R2E-Gym.git
```

`swebench` provides the grading harness (`swebench.harness.*`) and `r2egym`
provides the trajectory utilities and log parsers used by the R2E-Gym reward.
Both are imported lazily, so the rest of AgentFly works without them, but any
SWE reward call will fail with `ImportError` until they are installed.

## 2. enroot

The SWE rewards launch sandboxed shells inside enroot containers. Install the
enroot system package as described in
[Installation](../start/installation.md#requirements) — sudo is required.

## 3. Fetch the container images

Each task uses a different image set. Place all of them under one directory and
remember the path:

| Task | Image set |
|---|---|
| SWE-bench Verified evaluation | `swe-bench-verified` |
| R2E-Gym training (lite) | `r2e-gym-lite` |

Example layout:

```
/some/big/disk/enroot/images/
├── r2e-gym-lite/
│   ├── <instance>.sqsh
│   └── ...
└── swe-bench-verified/
    ├── <instance>.sqsh
    └── ...
```

The build/import commands depend on which image source you use (Docker Hub,
your registry, or pre-built `.sqsh` files); refer to the upstream R2E-Gym and
SWE-bench docs for image provisioning. AgentFly only needs the resulting
directory to exist and contain the per-instance images.

## 4. Set `ENROOT_IMAGES_PATH`

The example scripts and tests read `ENROOT_IMAGES_PATH` from the environment
and fall back to a placeholder otherwise. Export it in your shell before
launching, e.g.:

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

## 5. Smoke test

With the environment variable set and images present:

```bash
pytest tests/unit/rewards/swe/test_eval_r2e_gym.py -x
```

A successful run confirms that `r2egym`, `swebench`, enroot, and your image
directory are all wired up correctly.
