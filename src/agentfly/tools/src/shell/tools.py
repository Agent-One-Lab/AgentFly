"""
Shell command tool: acquires the container from context and runs a shell command.
Same pattern as file tools (context.metadata['image_id'], acquire_resource, run_cmd).
"""

import asyncio
import logging

from enroot.errors import (
    MemGuardError,
    MemoryError as EnrootMemoryError,
    TimeoutError as EnrootTimeoutError,
)
from ray.exceptions import RayTaskError

from ....core import Context
from ....resources import ContainerResourceSpec
from ...decorator import tool
from .command_filter import CommandFilter

WORKSPACE_DIR = "/testbed"

logger = logging.getLogger(__name__)

# Block dangerous / heavy commands (see command_filter defaults); allow_list bypasses blocks.
_SHELL_COMMAND_FILTER = CommandFilter(
    allow_list=[
        r"pip3?\s+install\s+pytest\b",
        r"pip3?\s+install\s+-e\s+\.",  # editable install in current dir (e.g. pip install -e .)
    ],
)


def _log_enroot_memory(cmd: str, inner: Exception) -> None:
    logger.error(
        "enroot memory error (%s): %s; command=%r",
        type(inner).__name__,
        inner,
        cmd,
    )


def _ensure_str(output) -> str:
    """Return output as str; decode from bytes if necessary."""
    if isinstance(output, bytes):
        return output.decode("utf-8")
    return output if output is not None else ""


async def _run_shell(context: Context, cmd: str) -> str:
    """
    Acquire container from context and run the given shell command.

    Args:
        context: The context object.
        cmd: The shell command to run.
        timeout: Timeout in seconds for the shell command inside the container.
    Returns:
        The output of the shell command, or a short error string for timeout / enroot memory limits.
    Other ``run_cmd`` failures propagate unchanged.
    """
    image_id = context.metadata.get("image_id")
    if not image_id:
        return "Error: context.metadata['image_id'] is required for shell tool."

    allowed, block_reason = _SHELL_COMMAND_FILTER.check(cmd)
    if not allowed:
        return block_reason

    rollout_id = context.rollout_id
    spec = ContainerResourceSpec(
        category="container",
        image=image_id,
    )
    is_acquired = context.is_spec_acquired(spec)

    container = await context.acquire_resource(
        id=rollout_id,
        spec=spec,
        backend=context.resource_backend,
        timeout=1200,
    )

    if not is_acquired:
        # Used for swe-smith, we need to checkout the specific commit first.
        # Use a bounded timeout so git operations cannot hang indefinitely.
        commit = context.metadata.get("git_commit_hash", None)
        if commit is not None:
            git_timeout = 300
            await container.run_cmd(
                "git fetch", timeout=git_timeout, workdir=WORKSPACE_DIR
            )
            await container.run_cmd(
                f"git checkout {commit}",
                timeout=git_timeout,
                workdir=WORKSPACE_DIR,
            )

    # We limit the memory to 1g for the shell command to avoid the container being killed by the system.
    try:
        raw = await container.run_cmd(
            cmd,
            timeout=120,
            workdir=WORKSPACE_DIR,
            mem_limit="4g",
            mem_guard_limit="4g",
        )
    except (asyncio.TimeoutError, EnrootTimeoutError):
        return "Error: Command timed out."
    except (EnrootMemoryError, MemGuardError) as e:
        _log_enroot_memory(cmd, e)
        return "Error: Command exceeded memory limit."
    except Exception:
        raise
    return _ensure_str(raw)


@tool(name="run_shell_command", max_length=10000)
async def run_shell_command(cmd: str, context: Context):
    """
    Runs a shell command in the container workspace.

    Commands are checked by :class:`~.command_filter.CommandFilter` before execution;
    blocked commands return the filter reason (e.g. ``Blocked: ...``) without running.

    Args:
        cmd (str): The shell command to run (e.g. "ls -la", "cat file.txt", "pwd").
            For multi-line python -c code, use bash ANSI-C quoting so newlines work:
            python3 -c $'line1\\nline2\\nline3' (the \\n are real newlines inside the container).
        context (Context): Injected rollout context; used to acquire the container resource.
    """
    return await _run_shell(context, cmd)
