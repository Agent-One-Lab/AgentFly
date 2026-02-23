"""Tests for read_file tool with real container."""

import pytest

from agentfly.core import Context
from agentfly.tools.src.file.tools import list_files, read_file

IMAGE_ID = "swebench/swesmith.x86_64.andialbrecht_1776_sqlparse.e57923b3"


@pytest.mark.asyncio
async def test_read_file_existing():
    context = Context(
        rollout_id="test_read_file",
        group_id="group_file_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        listing = await list_files(path=".", context=context)
        paths = [p.strip() for p in listing['observation'].split("\n") if p.strip()]
        assert len(paths) > 0, "workspace should have at least one file"
        first_path = paths[0]
        result = await read_file(path=first_path, context=context)
        print(result)
    finally:
        await context.release_resource(scope="rollout")


@pytest.mark.asyncio
async def test_read_file_nonexistent_returns_error():
    context = Context(
        rollout_id="test_read_file_nonexistent",
        group_id="group_file_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        result = await read_file(path="README.rst", context=context)
        print(result)
    finally:
        await context.release_resource(scope="rollout")
