"""Tests for list_files tool with real container."""

import pytest

from agentfly.core import Context
from agentfly.tools.src.file.tools import list_files
from agentfly.tools.types import ToolResult

IMAGE_ID = "swebench/swesmith.x86_64.andialbrecht_1776_sqlparse.e57923b3"



@pytest.mark.asyncio
async def test_list_files_root():
    context = Context(
        rollout_id="test_list_files_root",
        group_id="group_file_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        result = await list_files(path=".", context=context)
        assert isinstance(result, ToolResult), f"expected ToolResult, got {type(result)}"
        obs = result.observation
        assert obs, "empty observation"
        print(obs)
    finally:
        await context.end_resource(scope="rollout")


@pytest.mark.asyncio
async def test_list_files_path_slash():
    """path='/' is normalized to workspace root (same as '.')."""
    context = Context(
        rollout_id="test_list_files_slash",
        group_id="group_file_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        dot = await list_files(path=".", context=context)
        slash = await list_files(path="/", context=context)
        d_obs, s_obs = dot.observation, slash.observation
        assert d_obs == s_obs, "path='/' should list the same workspace root as '.'"
        print(d_obs)
        print(s_obs)
    finally:
        await context.end_resource(scope="rollout")
