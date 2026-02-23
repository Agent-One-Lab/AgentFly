"""
File operation tools: module is mounted at INSTALL_PATH in the container;
each tool acquires the container and runs file_manager.py with the corresponding tool name and params.
"""

import json
import os

from ....core import Context
from ....resources import ResourceSpec
from ...decorator import tool

# Path inside the container where the file module is mounted
INSTALL_PATH = "/usr/local/bin"
# Host path to this module (for mount source)
FILE_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _escape_shell_json(payload: str) -> str:
    """Escape a JSON string for safe use inside single-quoted shell argument."""
    return payload.replace("'", "'\"'\"'")


async def _run_file_tool(context: Context, tool_name: str, params: dict) -> str:
    """
    Acquire container with file module mounted at INSTALL_PATH, run file_manager.py with tool name and params.
    """
    image_id = context.metadata.get("image_id")
    if not image_id:
        return "Error: context.metadata['image_id'] is required for file tools."
    rollout_id = context.rollout_id
    spec = ResourceSpec(
        category="container",
        image=image_id,
        mount={FILE_MODULE_DIR: f"{INSTALL_PATH}:ro,rbind"},
    )
    container = await context.acquire_resource(id=rollout_id, spec=spec)
    payload = json.dumps({"tool": tool_name, "params": params})
    escaped = _escape_shell_json(payload)
    cmd = f"python {INSTALL_PATH}/file_manager.py '{escaped}'"
    return (await container.run_cmd(cmd)).decode("utf-8")


@tool(name="read_file")
async def read_file(path: str, context: Context):
    """
    Reads a file from the workspace with line numbers.
    Args:
        path: The relative path to the file (under container workspace).
    """
    return await _run_file_tool(context, "read_file", {"path": path})


@tool(name="edit_file")
async def edit_file(
    path: str, search_block: str, replace_block: str, context: Context
):
    """
    Surgically replaces a block of text in a file. Only replaces the first occurrence of search_block.
    Args:
        path: Path to file.
        search_block: Exact text to find.
        replace_block: Text to insert.
    """
    return await _run_file_tool(
        context,
        "edit_file",
        {
            "path": path,
            "search_block": search_block,
            "replace_block": replace_block,
        },
    )


@tool(name="list_files")
async def list_files(path: str = ".", context: Context = None):
    """
    Lists all files recursively in the workspace.
    Args:
        path: Directory to start from (defaults to root).
    """
    return await _run_file_tool(context, "list_files", {"path": path})


@tool(name="grep_search")
async def grep_search(
    pattern: str, path: str = ".", context: Context = None
):
    """
    Search for a regex pattern across all files in a directory.
    Args:
        pattern: The regex/string to search for.
        path: Directory to search in.
    """
    return await _run_file_tool(
        context, "grep_search", {"pattern": pattern, "path": path}
    )


@tool(name="undo_edit")
async def undo_edit(path: str, context: Context):
    """
    Reverts the last modification made to a specific file.
    Args:
        path: The path of the file to revert.
    """
    return await _run_file_tool(context, "undo_edit", {"path": path})
