"""
Shell command tool: acquires the container from context and runs a shell command.
Same pattern as file tools (context.metadata['image_id'], acquire_resource, run_cmd).
"""

import asyncio

from ....core import Context
from ....resources import ResourceSpec
from ...decorator import tool

WORKSPACE_DIR = "/testbed"


def _ensure_str(output) -> str:
    """Return output as str; decode from bytes if necessary."""
    if isinstance(output, bytes):
        return output.decode("utf-8")
    return output if output is not None else ""


async def _run_shell(context: Context, cmd: str, timeout: int = 120) -> str:
    """
    Acquire container from context and run the given shell command.

    Args:
        context: The context object.
        cmd: The shell command to run.
        timeout: Timeout in seconds for the shell command inside the container.
    Returns:
        The output of the shell command.
    Raises:
        asyncio.TimeoutError: If the command timed out.
    """
    image_id = context.metadata.get("image_id")
    if not image_id:
        return "Error: context.metadata['image_id'] is required for shell tool."
    rollout_id = context.rollout_id
    spec = ResourceSpec(
        category="container",
        image=image_id,
    )
    is_acquired = context.is_spec_acquired(spec)

    container = await context.acquire_resource(id=rollout_id, spec=spec)

    if not is_acquired:
        # Used for swe-smith, we need to checkout the specific commit first.
        # Use a bounded timeout so git operations cannot hang indefinitely.
        commit = context.metadata.get("git_commit_hash", None)
        if commit is not None:
            git_timeout = max(timeout, 300)
            await container.run_cmd(
                "git fetch", timeout=git_timeout, workdir=WORKSPACE_DIR
            )
            await container.run_cmd(
                f"git checkout {commit}",
                timeout=git_timeout,
                workdir=WORKSPACE_DIR,
            )
    try:
        raw = await container.run_cmd(cmd, timeout=timeout, workdir=WORKSPACE_DIR)
    except asyncio.TimeoutError:
        return "Error: Command timed out."
    return _ensure_str(raw)


@tool(name="run_shell_command", max_length=10000)
async def run_shell_command(cmd: str, context: Context, timeout: int = 120):
    """
    Runs a shell command in the container workspace.

    Args:
        cmd: The shell command to run (e.g. "ls -la", "cat file.txt", "pwd").
            For multi-line python -c code, use bash ANSI-C quoting so newlines work:
            python3 -c $'line1\\nline2\\nline3' (the \\n are real newlines inside the container).
        timeout: Timeout in seconds for the shell command inside the container.
    """
    return await _run_shell(context, cmd, timeout=timeout)
