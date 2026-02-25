# swesmith_enroot

Use **installed** SWE-smith with **Enroot** instead of Docker for running evaluation. No changes to the SWE-smith codebase; this package patches the container layer at import time.

## Requirements

- [Enroot](https://github.com/NVIDIA/enroot) installed
- **enroot-py** (path `../enroot-py/src` or installable)
- **swesmith** and **swebench** installed (e.g. `pip install -e /path/to/SWE-smith` and swebench)

## Usage

Import **before** any swesmith or harness code:

```python
import swesmith_enroot  # MUST be first
from swesmith.harness.eval import main
main(run_id="test", workers=1, dataset_path="SWE-bench/SWE-smith-py", instance_ids=["..."])
```

Or run the one-example test (from repo root, with `test_swe` on `PYTHONPATH`):

```bash
PYTHONPATH=test_swe python test_swe/test_one_example.py
```

Edit `test_one_example.py` to import `swesmith_enroot` first, then use `from swesmith.harness.eval import main` (and remove the registry patching for the extracted swe_smith).

## What gets patched

1. **`docker` module** — `docker.from_env()`, `client.containers.create()`, `container.start()`, `container.exec_run(workdir=, user=, environment=)`, `container.put_archive()`, `container.kill()` are implemented with enroot-py.
2. **swebench.harness.docker_utils** — `copy_to_container`, `exec_run_with_timeout`, `cleanup_container` use the Enroot container adapter.
3. **swesmith.profiles.base.RepoProfile** — `pull_image` and `_cache_image_exists` use Enroot’s image pull/exists so no `docker pull` or `docker image inspect` subprocess is used.

## What stays Docker-only

- **build_image()** and **push_image()** are unchanged. Build images on a host that has Docker; evaluation only needs pull + run, which use Enroot.
