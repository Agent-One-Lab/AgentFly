"""Tests for get_patch_from_runtime with real container."""
import pytest
import os
os.environ["ENROOT_IMAGES_PATH"] = "/mnt/weka/home/renxi.wang/Agent-One-Lab/enroot-py/data/images"
os.environ["ENROOT_ASYNC"] = "1"
from agentfly.core import Context
from agentfly.resources import ContainerResourceSpec
from agentfly.rewards.swe_rewards.utils import get_patch_from_runtime


IMAGE_ID = "swebench/swesmith.x86_64.andialbrecht_1776_sqlparse.e57923b3"


@pytest.mark.asyncio
async def test_get_patch_from_runtime_swe_smith():
    """With dataset swe-smith and workspace /testbed, get_patch returns str or None."""
    context = Context(
        rollout_id="test_get_patch",
        group_id="group_swe_reward",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        container = await context.acquire_resource(
            spec=ContainerResourceSpec(category="container", image=IMAGE_ID),
            id=context.rollout_id,
            scope="rollout",
            backend="local",
        )
        result = await get_patch_from_runtime(
            container,
            dataset="swe-smith",
            workspace_dir_name="/testbed",
            timeout=120,
        )
        print(f"Evaluation result with no edits: {result}")
        # No edits in a fresh workspace: empty diff or empty string; or a valid diff
        assert result is None or isinstance(result, str)
        if result:
            # If we got content, it should look like a diff or be cleaned
            assert isinstance(result, str)
    finally:
        await context.release_resource(scope="rollout")


@pytest.mark.asyncio
async def test_get_patch_from_runtime_with_instance_base_commit():
    """With instance.base_commit and no swe-smith dataset, edits produce a non-empty diff."""
    context = Context(
        rollout_id="test_get_patch_base",
        group_id="group_swe_reward",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        container = await context.acquire_resource(
            spec=ContainerResourceSpec(category="container", image=IMAGE_ID),
            id=context.rollout_id,
            scope="rollout",
            backend="local",
        )

        # Pick a tracked file from the repo and append a unique marker line
        tracked = await container.run_cmd(
            "cd /testbed && git ls-files | head -n 1",
            timeout=60,
        )
        tracked = tracked.strip().splitlines()[0]
        marker = "# test_get_patch_from_runtime_marker"
        await container.run_cmd(
            f"cd /testbed && echo '{marker}' >> {tracked}",
            timeout=60,
        )

        result = await get_patch_from_runtime(
            container,
            instance={"base_commit": "HEAD"},
            dataset="other",
            workspace_dir_name="/testbed",
            timeout=120,
        )
        print(f"Evaluation result with edits: {result}")
        assert isinstance(result, str)
        assert marker in result
    finally:
        await context.release_resource(scope="rollout")
