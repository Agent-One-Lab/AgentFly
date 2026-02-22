"""
ScienceWorld environment as a ContainerResource.

The runner starts the container; this class connects via HTTP and provides step/reset.
Acquire via Context: await context.acquire_resource(spec=ScienceWorldSpec, scope="rollout", backend="local").
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Union

import httpx

from ..resources import ContainerResource, ResourceSpec


ScienceWorldSpec = ResourceSpec(
    category="scienceworld",
    image="rifoag/scienceworld-env:latest",
    ports={"2700/tcp": None},
    extra={
        "container_port": 2700,
        "start_timeout": 10.0,
        "host_ip": "127.0.0.1",
    },
    max_global_num=24,
)


class ScienceWorldEnv(ContainerResource):
    """
    ContainerResource for ScienceWorld. Runner starts the container; start() connects and waits for /health.
    Use via Context.acquire_resource(spec=ScienceWorldSpec, scope="rollout", backend="local").
    """

    def __init__(self, container: Any, resource_id: str, spec: ResourceSpec):
        super().__init__(container=container, resource_id=resource_id, spec=spec)
        extra = spec.extra or {}
        self._container_port = int(extra.get("container_port", 2700))
        self._start_timeout = float(extra.get("start_timeout", 10.0))
        self._host_ip = extra.get("host_ip", "127.0.0.1")
        self._client: httpx.AsyncClient | None = None
        self.score = 0

    @property
    def category(self) -> str:
        return "scienceworld"

    async def _connect(self) -> None:
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
        self._client = httpx.AsyncClient(base_url=base_url, timeout=40.0)

    async def _wait_ready(self) -> None:
        deadline = time.time() + self._start_timeout
        while time.time() < deadline:
            try:
                if self._client and (await self._client.get("/health")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.1)
        raise RuntimeError("ScienceWorld did not become ready within timeout.")

    async def start(self) -> None:
        await super().start()
        await self._connect()
        await self._wait_ready()
        await self._client.get("/load?task_name=boil&variation_idx=0")

    async def reset(self, env_args: Any = None) -> str:
        env_args = env_args or {}
        task_name = env_args.get("task_name", "boil")
        variation_idx = env_args.get("variation_idx", 0)
        r = await self._client.get(f"/load?task_name={task_name}&variation_idx={variation_idx}")
        await self._client.get("/reset")
        data = r.json()
        self.score = 0
        return data.get("observation", "")

    async def step(self, action: str) -> Union[str, dict]:
        if action == "get_reward":
            return {
                "observation": "Task completed" if self.score >= 1 else "Task not completed",
                "reward": self.score,
            }
        r = await self._client.get(f"/step?action={action}")
        data = r.json()
        current_reward = data.get("info", {}).get("score", 0) / 100
        self.score = max(self.score, current_reward)
        return data.get("observation", "")

    async def end(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        await super().end()
