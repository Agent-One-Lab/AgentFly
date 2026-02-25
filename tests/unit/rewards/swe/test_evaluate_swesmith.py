"""Tests for evaluate_swesmith with real container.

Requires swesmith/swebench and the Enroot patch (import swe_rewards.swesmith_patch first).
The sample must match the image (instance_id / repo) so the profile can be resolved.
"""

import pytest

from agentfly.core import Context
from agentfly.resources import ResourceSpec

# Ensure Enroot patch is applied before any swesmith/swebench code
pytest.importorskip("swesmith")
pytest.importorskip("swebench")
from agentfly.rewards.swe_rewards.swesmith_patch import evaluate_swesmith

IMAGE_ID = "swebench/swesmith.x86_64.andialbrecht_1776_sqlparse.e57923b3"


def _minimal_sample_for_sqlparse():
    """Minimal instance dict for the andialbrecht_1776_sqlparse image.
    Must contain KEY_INSTANCE_ID and keys required by registry.get_from_inst (e.g. repo).
    """
    try:
        from swebench.harness.constants import KEY_INSTANCE_ID
        from swesmith.constants import KEY_PATCH
    except ImportError:
        return None
    # instance_id format for SWE-smith often matches the image slug
    instance_id = "andialbrecht_1776_sqlparse"
    return {
        KEY_INSTANCE_ID: instance_id,
        "repo": "andialbrecht/sqlparse",  # or whatever the profile expects
        KEY_PATCH: None,  # gold patch; use None to rely on patch= argument
    }


@pytest.mark.asyncio
async def test_evaluate_swesmith_returns_status_and_resolved():
    """evaluate_swesmith with container returns dict with 'status' and 'resolved'."""
    sample = _minimal_sample_for_sqlparse()
    if sample is None:
        pytest.skip("swebench/swesmith constants not available")

    context = Context(
        rollout_id="test_eval_swesmith",
        group_id="group_swe_reward",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        container_res = await context.acquire_resource(
            spec=ResourceSpec(category="container", image=IMAGE_ID),
            id=context.rollout_id,
            scope="rollout",
            backend="local",
        )
        # Empty patch: should evaluate and return completed/error, not crash
        result = evaluate_swesmith(
            sample=sample,
            patch="",  # empty patch
            container=container_res._container,
        )
        assert isinstance(result, dict)
        assert "status" in result
        assert "resolved" in result
        assert result["status"] in ("timeout", "error", "completed")
        assert isinstance(result["resolved"], bool)
    except Exception as e:
        # Profile may not exist for this instance_id; skip if so
        if "get_from_inst" in str(e) or "instance" in str(e).lower() or "profile" in str(e).lower():
            pytest.skip(f"Sample not valid for this image: {e}")
        raise
    finally:
        await context.release_resource(scope="rollout")
