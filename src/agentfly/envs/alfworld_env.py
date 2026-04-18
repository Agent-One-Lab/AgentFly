"""
ALFWorld environment as a ContainerResource.

The runner starts the container; this class connects via HTTP and provides step/reset.
Acquire via Context: await context.acquire_resource(spec=ALFWorldSpec, scope="rollout", backend="local").
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..resources import ContainerResource, ContainerResourceSpec


ALFWorldSpec = ContainerResourceSpec(
    category="alfworld",
    image="bitalov/alfworld-http-env-3:latest",
    ports={"8000/tcp": None},
    environment={
        "ALFWORLD_DATA": "/root/.cache/alfworld",
        "TRAIN_EVAL": "train",
        "BATCH_SIZE": "1",
    },
    container_port=8000,
    start_timeout=120.0,
    host_ip="127.0.0.1",
    max_global_num=8,
)


class ALFWorldEnv(ContainerResource):
    """
    ContainerResource for ALFWorld. Runner starts the container; start() connects and waits for /health.
    Use via Context.acquire_resource(spec=ALFWorldSpec, scope="rollout", backend="local").
    """

    def __init__(self, container: Any, resource_id: str, spec: ContainerResourceSpec):
        super().__init__(container=container, resource_id=resource_id, spec=spec)
        self._container_port = int(spec.container_port or 8000)
        self._start_timeout = float(spec.start_timeout or 120.0)
        self._host_ip = spec.host_ip or "127.0.0.1"
        self._client: httpx.AsyncClient | None = None
        self._current_info: Optional[Dict[str, Any]] = None
        self._current_obs: Optional[str] = None

    @property
    def category(self) -> str:
        return "alfworld"

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
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def _wait_ready(self) -> None:
        deadline = time.time() + self._start_timeout
        while time.time() < deadline:
            try:
                if self._client and (await self._client.get("/health")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.1)
        raise RuntimeError("ALFWorld did not become ready within timeout.")

    async def start(self) -> None:
        await super().start()
        await self._connect()
        await self._wait_ready()

    async def reset(
        self, env_args: Optional[Dict[str, Any]] = None, split: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        env_args = env_args or {"task_id": "trial_T20190909_013611_626994"}
        task_id = env_args.get("task_id", "trial_T20190909_013611_626994")
        split = split or "train"
        reset_data = {"split": split, "task_id": task_id}
        resp = await self._client.post("/reset", json=reset_data)
        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(f"Reset failed: {data}")
        self._current_obs = data["observation"]
        self._current_info = data.get("info", {})
        return data["observation"], self._current_info

    async def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        resp = await self._client.post("/step", json={"action": action})
        data = resp.json()
        if resp.status_code == 200:
            obs = data["observation"]
            reward = data.get("reward", 0.0)
            done = data.get("done", False)
            info = data.get("info", {})
            if self._current_info and "goal" in self._current_info and "goal" not in info:
                info["goal"] = self._current_info["goal"]
            self._current_info = info
            self._current_obs = obs
            return obs, reward, done, info
        return data.get("detail", "Unknown error"), 0.0, False, {}

    async def get_admissible_commands(self) -> List[str]:
        try:
            resp = await self._client.get("/admissible_commands")
            data = resp.json()
            return data.get("commands", []) if resp.status_code == 200 else []
        except Exception:
            return []

    async def get_info(self) -> Dict[str, Any]:
        if self._current_info:
            task_info = self._current_info.get("extra.gamefile", self._current_info.get("task", "unknown"))
            task = task_info[0] if isinstance(task_info, list) and task_info else task_info
            return {
                "task": task,
                "goal": self._current_info.get("goal") or self._current_info.get("task_description") or "",
                "won": self._current_info.get("won", False),
                "lost": self._current_info.get("lost", False),
                "admissible_commands_count": len(self._current_info.get("admissible_commands", [])),
            }
        try:
            resp = await self._client.get("/info")
            resp.raise_for_status()
            return resp.json().get("info", {})
        except (httpx.RequestError, httpx.HTTPStatusError):
            return {}

    async def end(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        await super().end()
