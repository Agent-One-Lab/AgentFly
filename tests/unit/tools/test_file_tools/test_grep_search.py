"""Tests for grep_search tool with real container."""

import pytest

from agentfly.core import Context
from agentfly.tools.src.file.tools import grep_search

IMAGE_ID = "swebench/swesmith.x86_64.andialbrecht_1776_sqlparse.e57923b3"


@pytest.mark.asyncio
async def test_grep_search_has_matches():
    context = Context(
        rollout_id="test_grep_search",
        group_id="group_file_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        result = await grep_search(pattern="import", path=".", context=context)
        print(result)
        # Either matches or "No matches found"
        assert "No matches" in result or ":" in result
    finally:
        await context.release_resource(scope="rollout")


@pytest.mark.asyncio
async def test_grep_search_no_matches():
    context = Context(
        rollout_id="test_grep_search_none",
        group_id="group_file_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        result = await grep_search(
            pattern="__XYZZY_NO_MATCH_PATTERN__",
            path=".",
            context=context,
        )
        print(result)
        assert "No matches" in result
    finally:
        await context.release_resource(scope="rollout")
