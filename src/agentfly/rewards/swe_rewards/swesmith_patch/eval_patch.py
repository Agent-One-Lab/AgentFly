"""
Patch run_evaluation to accept an optional container so callers can reuse
an existing container (e.g. for reward computation).

Blocking behavior (run_evaluation is fully synchronous):
- run_patch_in_container() does multiple blocking container.exec_run() and
  copy_to_container() calls, and one long exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)
  that runs the test suite and can take minutes.
- After that, file I/O (open report_path, get_eval_report reading test log) is also sync.
So run_evaluation cannot be made async without rewriting the swebench/swesmith harness.
Callers must run it in asyncio.to_thread() (or a process pool) to avoid blocking the event loop.
"""

import json


def _patch_run_evaluation():
    from swebench.harness.constants import (
        KEY_INSTANCE_ID,
        KEY_MODEL,
        KEY_PREDICTION,
        LOG_REPORT,
        LOG_TEST_OUTPUT,
        RUN_EVALUATION_LOG_DIR,
    )
    from swebench.harness.docker_build import close_logger
    from swesmith.constants import KEY_TIMED_OUT
    from swesmith.harness import eval as eval_mod
    from swesmith.harness.grading import get_eval_report
    from swesmith.harness.utils import run_patch_in_container
    from swesmith.profiles import registry

    _original = eval_mod.run_evaluation

    def run_evaluation(
        pred,
        instance,
        run_id: str,
        f2p_only: bool = False,
        is_gold: bool = False,
        *,
        container=None,
    ):
        """
        Same as original, but when container is provided it is passed through
        to run_patch_in_container so no new container is created or cleaned up.
        """
        instance_id = pred[KEY_INSTANCE_ID]
        rp = registry.get_from_inst(instance)
        # Blocking: git checkouts, apply patch, then exec_run_with_timeout(run tests) for up to rp.timeout seconds
        logger, timed_out = run_patch_in_container(
            instance,
            run_id,
            RUN_EVALUATION_LOG_DIR,
            rp.timeout,
            patch=pred[KEY_PREDICTION],
            commit=instance_id,
            f2p_only=f2p_only,
            is_gold=is_gold,
            container=container,
        )

        eval_folder = RUN_EVALUATION_LOG_DIR / run_id
        report_path = eval_folder / instance_id / LOG_REPORT
        test_log_path = eval_folder / instance_id / LOG_TEST_OUTPUT

        if timed_out:
            logger.info(f"Timed out for {instance_id}.")
            with open(report_path, "w") as f:
                f.write(json.dumps({KEY_TIMED_OUT: True, "timeout": rp.timeout}, indent=4))
            close_logger(logger)
            return {"status": "timeout", "resolved": False}

        if not test_log_path.exists():
            logger.info(f"Failed to get report for {instance_id}.")
            close_logger(logger)
            return {"status": "error", "resolved": False}

        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(pred, instance, test_log_path, f2p_only=f2p_only)
        report[KEY_MODEL] = pred[KEY_MODEL]
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        close_logger(logger)
        resolved = report.get("resolved", False)
        return {"status": "completed", "resolved": resolved}

    eval_mod.run_evaluation = run_evaluation
