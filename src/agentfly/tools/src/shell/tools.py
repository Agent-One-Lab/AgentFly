"""
Shell command tool: acquires the container from context and runs a shell command.
Same pattern as file tools (context.metadata['image_id'], acquire_resource, run_cmd).
"""

import asyncio

from ....core import Context
from ....resources import ResourceSpec
from ...decorator import tool


def _ensure_str(output) -> str:
    """Return output as str; decode from bytes if necessary."""
    if isinstance(output, bytes):
        return output.decode("utf-8")
    return output if output is not None else ""


async def _run_shell(context: Context, cmd: str) -> str:
    """
    Acquire container from context and run the given shell command.
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
    if not is_acquired:
        print("Not acquired")

    container = await context.acquire_resource(id=rollout_id, spec=spec)

    if not is_acquired:
        commit = context.metadata.get("instance_id")
        await container.run_cmd("git fetch", workdir="/testbed")
        await container.run_cmd(f"git checkout {commit}", workdir="/testbed")
    try:
        raw = await container.run_cmd(cmd, timeout=300)
    except asyncio.TimeoutError:
        return "Error: Command timed out."
    return _ensure_str(raw)


@tool(name="run_shell_command", max_length=10000)
async def run_shell_command(cmd: str, context: Context):
    """
    Runs a shell command in the container workspace.
    Args:
        cmd: The shell command to run (e.g. "ls -la", "cat file.txt", "pwd").
    """
    return await _run_shell(context, cmd)
