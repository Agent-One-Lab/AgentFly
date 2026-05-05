"""Tests for undo_edit tool with real container."""

import pytest

from agentfly.core import Context
from agentfly.tools.src.file.tools import edit_file, list_files, read_file, undo_edit

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


@pytest.mark.asyncio(loop_scope="session")
async def test_undo_edit_reverts_last_edit():
    context = Context(
        rollout_id="test_undo_edit",
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
        lines = [ln for ln in content_before.split("\n") if ln.strip()]
        if not lines:
            pytest.skip("No non-empty lines to edit")
        search_block = lines[7]
        replace_block = search_block + "  # undo_test"

        edit_result = (await edit_file(
            path=path,
            search_block=search_block,
            replace_block=replace_block,
            context=context,
        ))['observation']
        print(f"edit_result: {edit_result}")
        if "Error" in edit_result:
            pytest.skip("Edit failed, cannot test undo")
        after_edit = (await read_file(path=path, context=context))['observation']
        assert replace_block in _strip_line_numbers(after_edit)

        undo_result = (await undo_edit(path=path, context=context))['observation']
        print(f"undo_result: {undo_result}")
        ur = undo_result.lower()
        assert "undo successful" in ur or "error" in ur

        after_undo = (await read_file(path=path, context=context))['observation']
        print(f"after_undo: {after_undo}")
        assert search_block in _strip_line_numbers(after_undo)
        assert replace_block not in _strip_line_numbers(after_undo)
    finally:
        await context.end_resource(scope="rollout")


@pytest.mark.asyncio(loop_scope="session")
async def test_undo_edit_no_backup_returns_error():
    context = Context(
        rollout_id="test_undo_edit_no_backup",
        group_id="group_file_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        listing = (await list_files(path=".", context=context))['observation']
        paths = [p.strip() for p in listing.split("\n") if p.strip()]
        assert len(paths) > 0
        path = paths[0]
        result = (await undo_edit(path=path, context=context))['observation']
        # No prior edit: file_manager returns "No edit history for this file."
        print(f"observation: {result}")
        rl = result.lower()
        assert "no edit history" in rl or "error" in rl
    finally:
        await context.end_resource(scope="rollout")
