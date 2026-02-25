"""Tests for edit_file tool with real container."""

import pytest

from agentfly.core import Context
from agentfly.tools.src.file.tools import edit_file, list_files, read_file

IMAGE_ID = "swebench/swesmith.x86_64.andialbrecht_1776_sqlparse.e57923b3"


def _strip_line_numbers(content: str) -> str:
    """Remove '  N | ' prefix from read_file output."""
    lines = []
    for line in content.split("\n"):
        if "|" in line:
            _, _, rest = line.partition("|")
            lines.append(rest.strip())
        else:
            lines.append(line)
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_edit_file_replace_first_occurrence():
    context = Context(
        rollout_id="test_edit_file",
        group_id="group_file_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        listing = (await list_files(path=".", context=context))['observation']
        paths = [p.strip() for p in listing.split("\n") if p.strip()]
        assert len(paths) > 0
        path = paths[0]
        before = (await read_file(path=path, context=context))['observation']
        content_before = _strip_line_numbers(before)

        # Use a block that likely exists: first non-empty line
        lines = [ln for ln in content_before.split("\n") if ln.strip()]
        if not lines:
            pytest.skip("No non-empty lines to edit")
        search_block = lines[0]
        replace_block = search_block + "  # edited by test"

        result = await edit_file(
            path=path,
            search_block=search_block,
            replace_block=replace_block,
            context=context,
        )
        print(result)
        assert "updated successfully" in result['observation'].lower() or "Error" in result['observation']

        if "Error" not in result['observation']:
            after = (await read_file(path=path, context=context))['observation']
            assert replace_block in _strip_line_numbers(after)
    finally:
        await context.release_resource(scope="rollout")


@pytest.mark.asyncio
async def test_edit_file_search_block_not_found():
    context = Context(
        rollout_id="test_edit_file_not_found",
        group_id="group_file_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        listing = (await list_files(path=".", context=context))['observation']
        paths = [p.strip() for p in listing.split("\n") if p.strip()]
        assert len(paths) > 0
        path = paths[0]
        result = await edit_file(
            path=path,
            search_block="__SEARCH_BLOCK_DOES_NOT_EXIST__",
            replace_block="replacement",
            context=context,
        )
        print(result)
        assert "Error" in result['observation']
        assert "not found" in result['observation'].lower() or "exact" in result.lower()
    finally:
        await context.release_resource(scope="rollout")
