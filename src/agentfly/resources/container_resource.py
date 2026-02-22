"""
Container resource implementation.

Wraps an enroot Container (from src/enroot) to implement BaseResource,
so the resource engine can manage containers via LocalRunner.
"""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

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

    async def run_cmd(self, cmd: str) -> str:
        # Run via shell so builtins (cd, &&, etc.) work; exec_run(cmd) would run "cd" as a program and fail
        result = await asyncio.to_thread(
            self._container.exec_run, ["sh", "-c", cmd]
        )
        return result.output if hasattr(result, "output") else str(result)
