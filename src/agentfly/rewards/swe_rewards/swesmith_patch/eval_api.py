"""
Evaluate a single sample (dataset instance) and return the result.

Use this after importing swesmith_enroot so evaluation runs with Enroot.
"""
from typing import Any


def evaluate_swesmith(
    sample: dict,
    patch: str | None = None,
    run_id: str = "eval",
    model_name: str = "model",
    f2p_only: bool = False,
    is_gold: bool = False,
    container: Any = None,
) -> dict:
    """
    Evaluate a single sample (dataset instance) and return the result.

    Args:
        sample: Instance dict from the dataset (must contain KEY_INSTANCE_ID and
            profile keys such as repo, FAIL_TO_PASS, PASS_TO_PASS; KEY_PATCH used if patch is None).
        patch: Fix patch (diff string) to evaluate. If None, uses sample[KEY_PATCH] (gold).
        run_id: Run identifier for log paths.
        model_name: Label for this prediction (e.g. "gold" or model name).
        f2p_only: If True, run only tests in files that have fail-to-pass tests.
        is_gold: If True, treat patch as the gold (bug-fix) patch and apply in reverse.

    Returns:
        dict with:
            - "status": "timeout" | "error" | "completed"
            - "resolved": bool (True iff the instance was resolved)
    """
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
    from swesmith.harness.eval import run_evaluation

    patch_to_use = patch
    # patch_to_use = sample.get(KEY_PATCH)
    instance_id = sample[KEY_INSTANCE_ID]
    pred = {
        KEY_INSTANCE_ID: instance_id,
        KEY_PREDICTION: patch_to_use,
        KEY_MODEL: model_name,
    }
    print(f"Pred: {pred}")
    return run_evaluation(
        pred, sample, run_id, f2p_only=f2p_only, is_gold=is_gold, container=container
    )
