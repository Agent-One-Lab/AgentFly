"""
Container resource implementation.

Wraps an enroot Container (from src/enroot) to implement BaseResource,
so the resource engine can manage containers via LocalRunner.
"""

from __future__ import annotations

import asyncio
import shlex
from typing import Any, Optional, TYPE_CHECKING

from .types import BaseResource, ResourceSpec, ResourceStatus



class ContainerResource(BaseResource):
    """Wraps an enroot Container to satisfy BaseResource."""

    def __init__(self, container: Any, resource_id: str, spec: ResourceSpec):
        self._container = container
        self._resource_id = resource_id
        self._spec = spec

    @property
    def resource_id(self) -> str:
        return self._resource_id

    @property
    def category(self) -> str:
        return "container"

    @property
    def container(self) -> Any:
        """Backend handle for exec_run etc. Used by call interfaces."""
        return self._container

    async def start(self) -> None:
        # Enroot container is already started by Containers.run
        await asyncio.to_thread(self._container.reload)

    async def get_status(self) -> ResourceStatus:
        def _reload_and_status():
            self._container.reload()
            s = getattr(self._container, "status", "exited")
            return s

        status = await asyncio.to_thread(_reload_and_status)
        if status == "running":
            return ResourceStatus.RUNNING
        if status == "exited":
            return ResourceStatus.STOPPED
        return ResourceStatus.PENDING

    async def control(self, **kwargs: Any) -> None:
        # Enroot/systemd-run limits are set at start; dynamic control is limited
        # Subclasses or backend could support cgroup updates if available
        pass

    async def close(self) -> None:
        await asyncio.to_thread(self._container.kill)

    async def reset(self) -> None:
        raise NotImplementedError("Basic ContainerResource does not support reset.")

    async def end(self) -> None:
        await asyncio.to_thread(self._container.kill)

    async def run_cmd(
        self,
        cmd: str,
        timeout: Optional[float] = None,
        *,
        workdir: Optional[str] = None,
        user: Optional[str] = None,
        environment: Optional[dict] = None,
        **exec_kwargs: Any,
    ) -> str:
        """
        Execute a shell command in the container.

        The command is run as ``bash -c "<cmd>"`` so that shell features (cd, &&, ||,
        pipes, redirects) work. Without this, exec_run would treat the first token
        as the executable (e.g. "cd" would be run as a program and fail, since cd
        is a shell builtin). Using bash allows ANSI-C quoting (e.g. ``$'line1\nline2'``)
        so that ``\\n`` is interpreted as a newline in multi-line python -c commands.

        Args:
            cmd: Shell command string (executed with bash -c).
            timeout: Optional timeout in seconds. If provided, the command is
                wrapped with the shell ``timeout`` utility so that it is
                terminated inside the container when the timeout elapses. When
                a timeout occurs, asyncio.TimeoutError is raised.
            workdir: Optional working directory inside the container for this exec.
            user: Optional user (name or uid) to run the command as.
            environment: Optional dict of env vars for this exec only.
            **exec_kwargs: Additional keyword arguments passed through to
                the container's exec_run (e.g. privileged, demux).

        Returns:
            Command stdout as string (decoded from bytes if needed).

        Raises:
            asyncio.TimeoutError: If timeout is set and the command is
                terminated by the shell ``timeout`` utility.
        """
        shell_cmd = cmd
        if timeout is not None:
            # timeout(1) runs a single COMMAND; it does not run a shell. So "timeout 5s cd /x && foo"
            # would try to exec "cd", which fails (cd is a builtin). Run the user command as one
            # shell invocation: timeout Ns bash -c '<quoted cmd>'.
            shell_cmd = f"timeout {timeout}s bash -c {shlex.quote(cmd)}"

        exec_args: list = ["bash", "-c", shell_cmd]
        kwargs: dict = {**exec_kwargs}
        if workdir is not None:
            kwargs["workdir"] = workdir
        if user is not None:
            kwargs["user"] = user
        if environment is not None:
            kwargs["environment"] = environment

        # Outer timeout: if the container/enroot exec_run never returns (e.g. container
        # frozen), we must not wait forever. When timeout is set, cap wall-clock wait
        # at timeout + 60s; otherwise use a large default so one stuck exec cannot hang
        # the step for hours.
        wall_timeout = (timeout + 60.0) if timeout is not None else 3600.0
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._container.exec_run, exec_args, **kwargs
                ),
                timeout=wall_timeout,
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Command did not complete within {wall_timeout}s (container may be unresponsive): {cmd}"
            )

        exit_code = getattr(result, "exit_code", None)
        if timeout is not None and exit_code == 124:
            raise asyncio.TimeoutError(
                f"Command timed out after {timeout} seconds: {cmd}"
            )

        output = result.output if hasattr(result, "output") else str(result)
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output if output is not None else ""
