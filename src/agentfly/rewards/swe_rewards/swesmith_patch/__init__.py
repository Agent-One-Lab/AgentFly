"""
Enroot backend for SWE-smith.

Import this module before any swesmith or swebench harness code so that
Docker is replaced by Enroot (enroot-py must be aligned with docker API; see ENROOT_DOCKER_ALIGNMENT.md).

Usage:
    import swesmith_enroot  # MUST be first
    from swesmith.harness.eval import main
    main(run_id="...", workers=1, ...)

    # Or evaluate a single sample and get the result:
    from swesmith_enroot import evaluate
    result = evaluate(sample)  # result["status"], result["resolved"]
"""

import sys
from pathlib import Path
from .utils_patch import _patch_run_patch_in_container



def _install():

    # Swebench docker_utils (exec_run_with_timeout, cleanup_container) use container.client.api
    # and container.stop(timeout=); enroot-py provides these, so no docker_utils patch needed.

    from . import profile_patch
    profile_patch._patch_repo_profile()

    from .utils_patch import _patch_run_patch_in_container
    _patch_run_patch_in_container()

    from .eval_patch import _patch_run_evaluation
    _patch_run_evaluation()



# Apply patch BEFORE importing eval_api, so swebench/swesmith see patched docker when they import it.
try:
    _install()
except ImportError as e:
    raise ImportError(
        "Enroot docker compatibility not found. Ensure enroot-py is installed and has "
        "enroot.docker_compat (see test_swe/ENROOT_DOCKER_ALIGNMENT.md)."
    ) from e

from .eval_api import evaluate_swesmith


__all__ = ["evaluate_swesmith"]
