"""
File operation tools: the module is copied (on demand, once per rollout) to
INSTALL_PATH in the container; each tool acquires the container and runs
file_manager.py with the corresponding tool name and params.
"""

import asyncio
import json
import os
from typing import Optional

from .... import AF_CONTAINER_TOOLS_DIR
from ....core import Context
from ....resources import ContainerResourceSpec
from ...decorator import tool

# Path inside the container where the file module is mounted. Lives under the
# single agentfly container home (AF_CONTAINER_TOOLS_DIR = $AF_CONTAINER_HOME/
# tools) alongside skills + other tool helpers. A dedicated dir (NOT
# /usr/local/bin) so the read-only bind mount can't shadow image binaries —
# e.g. python:slim keeps python3/pip in /usr/local/bin, and mounting over it
# makes `python3` vanish.
INSTALL_PATH = os.path.join(AF_CONTAINER_TOOLS_DIR, "file")
# Host path to this module (for mount source)
FILE_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _escape_shell_json(payload: str) -> str:
    """Escape a JSON string for safe use inside single-quoted shell argument."""
    return payload.replace("'", "'\"'\"'")


async def _run_file_tool(context: Context, tool_name: str, params: dict) -> str:
    """
    Acquire the rollout container, copy-on-demand the file module to INSTALL_PATH
    (once per rollout), then run file_manager.py with the tool name and params.
    """
    image_id = context.metadata.get("image_id")
    if not image_id:
        return "Error: context.metadata['image_id'] is required for file tools."
    rollout_id = context.rollout_id
    spec = ContainerResourceSpec(
        category="container",
        image=image_id,
        docker_host=context.metadata.get("docker_host"),
    )
    container = await context.acquire_resource(id=rollout_id, spec=spec)
    # Copy-on-demand: stage the file module into the container the first time a
    # file tool runs this rollout (no bind mount — works for skills from any
    # host root too, and avoids the "mounts fixed at first acquire" constraint).
    staged = context.metadata.setdefault("_staged_paths", set())
    if INSTALL_PATH not in staged:
        await container.copy_in(FILE_MODULE_DIR, INSTALL_PATH)
        staged.add(INSTALL_PATH)
    # Root the file manager at the rollout's WORKDIR (unified with the shell
    # tool). When set, it is passed explicitly and also becomes the exec cwd, so
    # file_manager.py's os.getcwd() fallback resolves to the same directory.
    workdir = context.metadata.get("workdir")
    payload = json.dumps({"tool": tool_name, "params": params, "workdir": workdir})
    escaped = _escape_shell_json(payload)
    # Many SWE images only provide ``python3``; merge stderr so failures are visible on stdout.
    cmd = (
        f"python3 {INSTALL_PATH}/file_manager.py '{escaped}' 2>&1 || "
        f"python {INSTALL_PATH}/file_manager.py '{escaped}' 2>&1"
    )
    try:
        raw = await container.run_cmd(cmd, timeout=120, workdir=workdir)
    except asyncio.TimeoutError:
        return f"{tool_name} timed out."
    out = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return out or ""


@tool(name="read_file", max_length=10000)
async def read_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None, context: Context = None):
    """
    Reads a file from the workspace with line numbers.
    Args:
        path: The relative path to the file (under container workspace).
        start_line: Start line number to read from.
        end_line: End line number to read to.
    """
    return await _run_file_tool(context, "read_file", {"path": path, "start_line": start_line, "end_line": end_line})


@tool(name="edit_file", max_length=10000)
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


@tool(name="list_files", max_length=10000)
async def list_files(path: str = ".", max_depth: int = 3, context: Context = None):
    """
    Lists all files recursively in the workspace.
    Args:
        path: Directory to start from (defaults to root).
        max_depth: Maximum depth to recurse into subdirectories.
    """
    return await _run_file_tool(context, "list_files", {"path": path, "max_depth": max_depth})


@tool(name="grep_search", max_length=10000)
async def grep_search(
    pattern: str, path: str = ".", include: str = "", context: Context = None
):
    """
    Search for a regex pattern across all files in a directory.
    Args:
        pattern: The regex/string to search for.
        path: Directory to search in.
        include: Glob filter, e.g. "*.py" for Python files only.
    """
    return await _run_file_tool(
        context, "grep_search", {"pattern": pattern, "path": path, "include": include}
    )


@tool(name="create_file", max_length=10000)
async def create_file(path: str, content: str = "", context: Context = None):
    """
    Create a new file under the workspace. Fails if the path already exists.
    Args:
        path: Relative path for the new file.
        content: Initial file body (default empty).
    """
    return await _run_file_tool(
        context, "create_file", {"path": path, "content": content}
    )


@tool(name="run_python", max_length=10000)
async def run_python(path: str, timeout: int = 60, context: Context = None):
    """
    Run Python only: executes a script file under the workspace (python3 path; cwd is workspace root).
    Args:
        path: Relative path to a .py file inside the workspace.
        timeout: Max seconds for the subprocess (default 60).
    """
    return await _run_file_tool(
        context, "python", {"path": path, "timeout": timeout}
    )


@tool(name="undo_edit", max_length=10000)
async def undo_edit(path: str, context: Context):
    """
    Reverts the last modification made to a specific file.
    Args:
        path: The path of the file to revert.
    """
    return await _run_file_tool(context, "undo_edit", {"path": path})
