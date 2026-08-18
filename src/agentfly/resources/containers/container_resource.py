"""
Container resource.

ContainerResource wraps a container handle (enroot OR docker — any object
exposing reload/status/exec_run[_async]/kill[_async]). Command execution
and lifecycle are unified here, independent of the backend engine.
"""

from __future__ import annotations

import asyncio
import base64
import shlex
from typing import Any, Optional, Tuple

from ..types import BaseResource, ContainerResourceSpec, ResourceStatus

# Treat the engine's own timeout type as a timeout when present, but don't
# hard-depend on enroot — the docker backend raises asyncio.TimeoutError.
_TIMEOUT_EXCS: Tuple[type, ...] = (asyncio.TimeoutError,)
try:  # pragma: no cover - enroot may be absent in some deployments
    from enroot.errors import TimeoutError as _EnrootTimeoutError
    _TIMEOUT_EXCS = (asyncio.TimeoutError, _EnrootTimeoutError)
except Exception:  # noqa: BLE001
    pass


class ContainerResource(BaseResource):
    """
    Local container-backed implementation of :class:`BaseResource`.

    `ContainerResource` wraps an enroot container handle and exposes the
    standard resource lifecycle (`start`, `get_status`, `end`, `close`) plus
    `run_cmd` for command execution inside the container.

    This class is the concrete resource returned by local container runners and
    is typically accessed via `Context.acquire_resource(...)` or `ResourceEngine.acquire(...)`.
    """

    def __init__(self, container: Any, resource_id: str, spec: ContainerResourceSpec):
        """
        Initialize a container resource wrapper.

        Args:
            container: Enroot container handle (must support reload/exec/kill operations).
            resource_id: Stable id used by the resource engine for tracking/reuse.
            spec: Resource specification that produced this resource.
        """
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
        """Refresh the container handle and ensure it is reachable."""
        await asyncio.to_thread(self._container.reload)

    async def get_status(self) -> ResourceStatus:
        """Return lifecycle status mapped from the underlying container status string."""
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
        """Update runtime limits if supported (currently a no-op for local enroot)."""
        pass

    async def close(self) -> None:
        """Alias for ending the resource by killing the underlying container."""
        if hasattr(self._container, "kill_async"):
            await self._container.kill_async()
        else:
            await asyncio.to_thread(self._container.kill)

    async def reset(self) -> None:
        """Reset is not supported for this resource type."""
        raise NotImplementedError("ContainerResource does not support reset.")

    async def end(self) -> None:
        """Terminate the underlying container and release its runtime handle."""
        if hasattr(self._container, "kill_async"):
            await self._container.kill_async()
        else:
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

        The command is passed via base64 and executed as ``echo <b64> | base64 -d | bash -s``,
        so the outer shell never parses the command string.

        Raises:
            asyncio.TimeoutError: If ``timeout`` is set and the backend ``exec_run`` times out.
        """
        b64 = base64.b64encode(cmd.encode("utf-8")).decode("ascii")
        shell_cmd = f"echo {shlex.quote(b64)} | base64 -d | bash -s"
        exec_args: list = ["bash", "-c", shell_cmd]
        kwargs: dict = {**exec_kwargs}
        if workdir is not None:
            kwargs["workdir"] = workdir
        if user is not None:
            kwargs["user"] = user
        if environment is not None:
            kwargs["environment"] = environment
        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            if hasattr(self._container, "exec_run_async"):
                result = await self._container.exec_run_async(exec_args, **kwargs)
            else:
                result = await asyncio.to_thread(
                    self._container.exec_run, exec_args, **kwargs
                )
        except _TIMEOUT_EXCS as e:
            if timeout is not None:
                raise asyncio.TimeoutError(
                    f"Command timed out after {timeout} seconds: {cmd}"
                ) from e
            raise asyncio.TimeoutError(str(e)) from e

        output = result.output if hasattr(result, "output") else str(result)
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output if output is not None else ""

    async def copy_in(self, local_path: str, container_path: str,
                      timeout: Optional[float] = 300) -> None:
        """Copy a local file/dir into the container (contents, for a dir).

        Distinct name from the backend handle's ``copy_to`` on purpose: a
        DockerContainer is its OWN handle, so a same-named method here would be
        shadowed by the sync handle ``copy_to``. Delegates to the handle's
        ``copy_to`` (enroot + docker both provide it), preferring an async
        variant. Used for copy-on-demand staging of tool helpers + skills.
        """
        handle = self._container
        # enroot's put_archive (unlike docker's) does NOT create the destination
        # dir before untarring, so tar fails with "Cannot open: No such file or
        # directory". Mirror DockerContainer.put_archive and mkdir -p it first.
        await self.run_cmd(f"mkdir -p {shlex.quote(container_path)}", timeout=60)
        if hasattr(handle, "copy_to_async"):
            await handle.copy_to_async(local_path, container_path, timeout)
        else:
            await asyncio.to_thread(handle.copy_to, local_path, container_path)
