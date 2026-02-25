"""
Patch run_patch_in_container to accept an optional existing container so callers
(e.g. reward computation) can reuse a container instead of launching a new one.
"""

import traceback
from pathlib import Path

from swebench.harness.constants import (
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    TESTS_TIMEOUT,
    UTF8,
)
from swebench.harness.docker_build import setup_logger
from swebench.harness.docker_utils import copy_to_container, exec_run_with_timeout
from swesmith.constants import LOG_DIR_RUN_VALIDATION, TEST_OUTPUT_END, TEST_OUTPUT_START
from swesmith.profiles import registry
from unidiff import PatchSet


def _patch_run_patch_in_container():
    from swesmith.harness import utils as utils_mod
    from swesmith.profiles.base import _find_ssh_key

    _apply_patch = utils_mod._apply_patch
    _ssh_copy_lock = utils_mod._ssh_copy_lock
    _original = utils_mod.run_patch_in_container

    def run_patch_in_container(
        instance,
        run_id: str,
        log_dir: Path,
        timeout: int,
        patch=None,
        commit=None,
        f2p_only: bool = False,
        is_gold: bool = False,
        *,
        container=None,
    ):
        """
        Same as original, but when `container` is provided: use that container
        (skip pull_image, create, start) and do not call cleanup_container at the end.
        Caller is responsible for the container lifecycle.
        """
        if container is None:
            return _original(
                instance, run_id, log_dir, timeout,
                patch=patch, commit=commit, f2p_only=f2p_only, is_gold=is_gold,
            )

        # Use existing container path
        instance_id = instance[KEY_INSTANCE_ID]
        rp = registry.get_from_inst(instance)
        is_eval = log_dir == RUN_EVALUATION_LOG_DIR
        container_type = "eval" if is_eval else "val"
        log_dir = log_dir / run_id / instance_id
        log_dir.mkdir(parents=True, exist_ok=True)
        container_name = f"swesmith.{container_type}.{run_id}.{instance_id}"
        log_file = log_dir / LOG_INSTANCE
        logger = setup_logger(container_name, log_file)

        ssh_env = {}
        if rp._is_repo_private():
            key_file = _find_ssh_key()
            if key_file is not None:
                with _ssh_copy_lock:
                    copy_to_container(container, key_file, Path("/github_key"))
                container.exec_run("chmod 600 /github_key", user=DOCKER_USER)
                ssh_env = {
                    "GIT_SSH_COMMAND": "ssh -i /github_key -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes"
                }

        try:
            if commit is not None:
                logger.info("Checking out commit %s", commit)
                # container.exec_run(
                #     "git fetch",
                #     workdir=DOCKER_WORKDIR,
                #     user=DOCKER_USER,
                #     environment=ssh_env,
                # )
                val = container.exec_run(
                    f"git checkout {commit}",
                    workdir=DOCKER_WORKDIR,
                    user=DOCKER_USER,
                )
                if val.exit_code != 0:
                    logger.info("CHECKOUT FAILED: %s", val.output.decode(UTF8))
                    return logger, False
                if is_eval:
                    val = container.exec_run(
                        "git checkout HEAD~1",
                        workdir=DOCKER_WORKDIR,
                        user=DOCKER_USER,
                    )
                    if val.exit_code != 0:
                        logger.info(
                            "CHECKOUT TO BUG STAGE FAILED: %s",
                            val.output.decode(UTF8),
                        )
                        return logger, False

            if patch is not None and len(patch) >= 1:
                logger.info("Applying patch to container...")
                changed_files = " ".join([x.path for x in PatchSet(patch)])
                container.exec_run(
                    f"git checkout -- {changed_files}",
                    workdir=DOCKER_WORKDIR,
                    user=DOCKER_USER,
                )
                patch_file = Path(log_dir / "patch.diff")
                patch_file.write_text(patch)
                copy_to_container(container, patch_file, Path(DOCKER_PATCH))
                _apply_patch(instance_id, container, logger, is_gold=is_gold)
                if is_eval:
                    f2p_files, p2p_files = rp.get_test_files(instance)
                    test_files = " ".join(f2p_files + p2p_files)
                    if test_files:
                        container.exec_run(
                            f"git checkout -- {test_files}",
                            workdir=DOCKER_WORKDIR,
                            user=DOCKER_USER,
                        )

            eval_file = Path(log_dir / "eval.sh")
            test_command, _ = rp.get_test_cmd(instance, f2p_only=f2p_only)
            eval_file.write_text(
                "\n".join(
                    [
                        "#!/bin/bash",
                        "set -uxo pipefail",
                        f"cd {DOCKER_WORKDIR}",
                        f": '{TEST_OUTPUT_START}'",
                        test_command,
                        f": '{TEST_OUTPUT_END}'",
                    ]
                )
                + "\n"
            )
            copy_to_container(container, eval_file, Path("/eval.sh"))

            # Main time sink: blocks for up to `timeout` seconds while tests run in container
            test_output, timed_out, total_runtime = exec_run_with_timeout(
                container, "/bin/bash /eval.sh", timeout=timeout
            )
            test_output_path = log_dir / LOG_TEST_OUTPUT
            logger.info("Test Runtime: %.2f seconds", total_runtime)
            with open(test_output_path, "w") as f:
                f.write(test_output)
                if timed_out:
                    f.write(f"\n\n{TESTS_TIMEOUT}: {timeout} seconds exceeded")
            return logger, timed_out
        except Exception as e:
            logger.info(
                "Error validating %s: %s\n%s",
                instance_id,
                e,
                traceback.format_exc(),
            )
            return logger, False

    utils_mod.run_patch_in_container = run_patch_in_container
