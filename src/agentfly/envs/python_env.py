"""
Python HTTP sandbox environment.

Subclasses ContainerResource to add HTTP /exec and /health behavior
for the reasonwang/python-http-env image. The runner starts the container
with port mapping; this class connects an httpx client and uses /exec for run_code.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
import httpx
from ..resources import ResourceSpec, ResourceCategory, ContainerResource

def python_sandbox_spec(
    image: str = "reasonwang/python-http-env:latest",
    container_port: int = 8000,
    start_timeout: float = 60.0,
    host_ip: str = "127.0.0.1",
) -> ResourceSpec:
    """ResourceSpec for the Python HTTP sandbox (container + HTTP /exec)."""
    return ResourceSpec(
        category=ResourceCategory.PYTHON_ENV,
        image=image,
        ports={f"{container_port}/tcp": None},
        extra={
            "python_http_env": True,
            "container_port": container_port,
            "start_timeout": start_timeout,
            "host_ip": host_ip,
        },
    )


class PythonSandboxEnv(ContainerResource):
    """
    ContainerResource that talks to the Python HTTP env image via /health and /exec.
    Expects the container to be started by the runner with port mapping; start()
    connects the httpx client and waits for /health.
    """

    def __init__(self, container: Any, resource_id: str, spec: ResourceSpec):
        super().__init__(container=container, resource_id=resource_id, spec=spec)
        extra = spec.extra or {}
        self._container_port = int(extra.get("container_port", 8000))
        self._start_timeout = float(extra.get("start_timeout", 60.0))
        self._host_ip = extra.get("host_ip", "127.0.0.1")
        self._client: httpx.AsyncClient | None = None
        self._episodes = 0

    async def _connect(self) -> None:
        """Discover host port and create httpx client to the container's HTTP server."""
        deadline = time.time() + self._start_timeout
        host_port = None
        while time.time() < deadline:
            await asyncio.to_thread(self._container.reload)
            ports = self._container.attrs.get("NetworkSettings", {}).get("Ports") or {}
            binding = ports.get(f"{self._container_port}/tcp")
            if binding and binding[0].get("HostPort"):
                host_port = binding[0]["HostPort"]
                break
            await asyncio.sleep(0.1)
        if host_port is None:
            try:
                logs = await asyncio.to_thread(self._container.logs)
                logs = logs.decode() if hasattr(logs, "decode") else str(logs)
            except Exception:
                logs = "Could not get logs"
            raise RuntimeError(f"Port mapping not found. {logs}")
        base_url = f"http://{self._host_ip}:{host_port}"
        self._client = httpx.AsyncClient(base_url=base_url, timeout=20.0)

    async def _wait_ready(self) -> None:
        """Poll /health until 200 or timeout."""
        deadline = time.time() + self._start_timeout
        while time.time() < deadline:
            try:
                if self._client:
                    r = await self._client.get("/health")
                    if r.status_code == 200:
                        return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.1)
        try:
            logs = await asyncio.to_thread(self._container.logs)
            logs = logs.decode() if hasattr(logs, "decode") else str(logs)
        except Exception:
            logs = "Could not get logs"
        raise RuntimeError(
            f"Sandbox did not become ready within {self._start_timeout}s.\n{logs}"
        )

    async def start(self) -> None:
        await super().start()
        await self._connect()
        await self._wait_ready()

    async def reset(self, *args: Any, **kwargs: Any) -> Any:
        try:
            await asyncio.wait_for(
                self.run_code("globals().clear()"), timeout=20.0
            )
        except (asyncio.TimeoutError, httpx.TransportError):
            await self.end()
            await super().start()
            await self._connect()
            await self._wait_ready()
        return ""

    async def step(self, code: str) -> str:
        """Execute code via HTTP POST /exec. Returns output or detail string."""
        if not self._client:
            raise RuntimeError("PythonSandboxEnv not started (no HTTP client)")
        self._episodes += 1
        resp = await self._client.post("/exec", json={"code": code})
        data = resp.json()
        if "output" in data:
            return data["output"]
        if "detail" in data:
            return data["detail"]
        raise RuntimeError(f"Unknown response: {data}")

    async def end(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        await super().end()

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        await super().close()
