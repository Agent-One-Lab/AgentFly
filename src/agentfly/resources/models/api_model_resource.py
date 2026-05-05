"""
OpenAI-compatible API model resource.

This resource connects to an already-running OpenAI-compatible API endpoint
instead of launching a new local process.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from ...utils.llm_backends.llm_backends import ClientBackend
from ..types import APIModelResourceSpec, BaseResource, ResourceStatus

logger = logging.getLogger(__name__)


class APIModelResource(BaseResource):
    """Managed wrapper for an existing OpenAI-compatible API service."""

    def __init__(
        self,
        spec: APIModelResourceSpec,
        resource_id: Optional[str] = None,
        *,
        startup_timeout: float = 60.0,
    ) -> None:
        self._spec = spec
        self._resource_id = resource_id or f"api-model-{uuid.uuid4().hex[:8]}"
        self._startup_timeout = startup_timeout
        self._status: ResourceStatus = ResourceStatus.PENDING
        self._backend: Optional[ClientBackend] = None

    @property
    def resource_id(self) -> str:
        return self._resource_id

    @property
    def category(self) -> str:
        return "api_model"

    @property
    def base_url(self) -> str:
        base_url = self._spec.base_url
        if base_url:
            return str(base_url).rstrip("/")

        host = str(self._spec.host or "127.0.0.1")
        port = int(self._spec.port or 8000)
        return f"http://{host}:{port}/v1"

    async def start(self) -> None:
        model_name_or_path = self._spec.model_name_or_path
        if not model_name_or_path:
            raise ValueError(
                "APIModelResourceSpec.model_name_or_path is required for category='api_model'."
            )

        self._backend = ClientBackend(
            model_name_or_path=model_name_or_path,
            base_url=self.base_url,
            api_key=str(self._spec.api_key or "EMPTY"),
            timeout=float(self._spec.request_timeout)
            if self._spec.request_timeout is not None
            else 3600,
        )
        await self._wait_until_ready()
        self._status = ResourceStatus.RUNNING

    async def _wait_until_ready(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._startup_timeout
        last_error: Optional[Exception] = None

        while loop.time() < deadline:
            try:
                await asyncio.to_thread(
                    urllib.request.urlopen,
                    f"{self.base_url}/models",
                    timeout=3.0,
                )
                return
            except Exception as e:  # service not ready yet
                last_error = e
                await asyncio.sleep(1.0)

        self._status = ResourceStatus.FAILED
        raise TimeoutError(
            f"Timed out waiting for API model endpoint at {self.base_url} to become ready."
        ) from last_error

    async def generate_async(
        self,
        messages: List[List[Dict[str, Any]]] | List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[str]:
        if self._backend is None:
            raise RuntimeError("API model resource is not started. Call start() first.")
        return await self._backend.generate_async(messages, **kwargs)

    async def reset(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("APIModelResource does not support reset.")

    async def get_status(self) -> ResourceStatus:
        return self._status

    async def control(self, **kwargs: Any) -> None:
        # Runtime control for remote API endpoints is not implemented.
        return None

    async def end(self) -> None:
        self._status = ResourceStatus.STOPPED
        self._backend = None

    async def close(self) -> None:
        await self.end()
