"""Tests for list_files tool with real container."""

import pytest

from agentfly.core import Context
from agentfly.tools.src.file.tools import list_files

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
        print(result)
        # Workspace should list at least one path (file or dir)
    finally:
        await context.release_resource(scope="rollout")


@pytest.mark.asyncio
async def test_list_files_path_slash():
    context = Context(
        rollout_id="test_list_files_slash",
        group_id="group_file_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        result = await list_files(path="/", context=context)
        print(result)
    finally:
        await context.release_resource(scope="rollout")
