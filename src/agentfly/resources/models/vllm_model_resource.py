"""
vLLM model resource.

This resource launches a local vLLM OpenAI-compatible service process and exposes
an async text generation API through :class:`ClientBackend`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import subprocess
import sys
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from ... import AF_DATA_DIR
from ...utils.llm_backends.llm_backends import ClientBackend
from ..types import BaseResource, ResourceStatus, VLLMModelResourceSpec

logger = logging.getLogger(__name__)


class VLLMModelResource(BaseResource):
    """Managed vLLM server resource with async generation support."""

    def __init__(
        self,
        spec: VLLMModelResourceSpec,
        resource_id: Optional[str] = None,
        *,
        startup_timeout: float = 300.0,
    ) -> None:
        self._spec = spec
        self._resource_id = resource_id or f"vllm-{uuid.uuid4().hex[:8]}"
        self._startup_timeout = startup_timeout
        self._process: Optional[subprocess.Popen] = None
        # vLLM's output goes here, so a failed start can say why.
        self._log_path = os.path.join(AF_DATA_DIR, "logs", f"{self._resource_id}.log")
        self._log_file = None
        self._status: ResourceStatus = ResourceStatus.PENDING
        self._backend: Optional[ClientBackend] = None

    @property
    def resource_id(self) -> str:
        return self._resource_id

    @property
    def category(self) -> str:
        return "vllm"

    @property
    def base_url(self) -> str:
        port = self._spec.port or 8000
        return f"http://127.0.0.1:{port}/v1"

    async def start(self) -> None:
        if self._process and self._process.poll() is None:
            self._status = ResourceStatus.RUNNING
            return

        model_name_or_path = self._spec.model_name_or_path
        if not model_name_or_path:
            raise ValueError("VLLMModelResourceSpec.model_name_or_path is required for category='vllm'.")

        tp = int(self._spec.tensor_parallel_size or 1)
        pp = int(self._spec.pipeline_parallel_size)
        dp = int(self._spec.data_parallel_size)
        template = self._spec.template
        gpu_memory_utilization = float(self._spec.gpu_memory_utilization)
        tool_call_parser = str(self._spec.tool_call_parser)
        port = int(self._spec.port or 8000)

        # Reuse the existing deployment entrypoint.
        cmd = [
            sys.executable,
            "-m",
            "agentfly.utils.deploy",
            "--model-name-or-path",
            model_name_or_path,
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--dp",
            str(dp),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--tool-call-parser",
            tool_call_parser,
        ]
        if template is not None:
            cmd.extend(["--template", str(template)])

        cmd.extend(["--port", str(port)])
        env = os.environ.copy()
        logger.info("[VLLMModelResource] Launching vLLM service: %s", " ".join(cmd))
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._log_file = open(self._log_path, "w")
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._backend = ClientBackend(
            model_name_or_path=model_name_or_path,
            base_url=self.base_url,
            api_key="EMPTY",
        )
        await self._wait_until_ready()
        self._status = ResourceStatus.RUNNING

    async def _wait_until_ready(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._startup_timeout
        last_error: Optional[Exception] = None

        while loop.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                self._status = ResourceStatus.FAILED
                raise RuntimeError(
                    f"vLLM service exited during startup with code {self._process.returncode}. "
                    f"Last lines of its log ({self._log_path}):\n{self._log_tail()}"
                )
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
            f"Timed out waiting for vLLM service at {self.base_url} to become ready (log: {self._log_path})."
        ) from last_error

    def _log_tail(self, lines: int = 20) -> str:
        try:
            with open(self._log_path, errors="replace") as f:
                return "".join(f.readlines()[-lines:])
        except OSError:
            return "(log unavailable)"

    async def generate_async(
        self,
        messages: List[List[Dict[str, Any]]] | List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[str]:
        if self._backend is None:
            raise RuntimeError("vLLM resource is not started. Call start() first.")
        return await self._backend.generate_async(messages, **kwargs)

    async def reset(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("VLLMModelResource does not support reset.")

    async def get_status(self) -> ResourceStatus:
        if self._process is None:
            return self._status
        if self._process.poll() is None:
            return ResourceStatus.RUNNING
        if self._status == ResourceStatus.RUNNING:
            return ResourceStatus.FAILED
        return self._status

    async def control(self, **kwargs: Any) -> None:
        # vLLM runtime control is not implemented yet.
        return None

    async def end(self) -> None:
        if self._process is None:
            self._status = ResourceStatus.STOPPED
            return

        if self._process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            try:
                await asyncio.to_thread(self._process.wait, 10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                await asyncio.to_thread(self._process.wait)

        self._status = ResourceStatus.STOPPED
        self._process = None
        self._backend = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    async def close(self) -> None:
        await self.end()
