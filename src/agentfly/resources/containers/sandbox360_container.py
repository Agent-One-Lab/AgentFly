"""Sandbox360 container backend: remote gVisor sandboxes over HTTP (LLM360/Sandbox360, docs/api.md).

Selected with ``container_engine="sandbox360"`` (or ``AF_CONTAINER_ENGINE=sandbox360``). Settings come from
``SANDBOX360_GATEWAY_URL`` and ``SANDBOX360_TOKEN``. A sandbox pulls the task's public image itself, so no image is
stored on this machine, and has no network unless the image's task asks for one. Its lease is renewed by one
background thread until the container is killed.

Retries follow the API's rule: a reply marked ``retryable`` means the command did not run. Anything else on an exec
(a lost stream, a read timeout, ``EXEC_TIMEOUT``) may have started it, so it is never resent.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import random
import tarfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import httpx

from .container_resource import ContainerResource

logger = logging.getLogger(__name__)

LEASE_SECONDS = 1800
RENEW_EVERY = 300  # the API's longest renewal interval
READY_POLL = 5.0
RETRIES = 8
EXEC_THREADS = 512  # blocking HTTP calls run here, so many episodes can wait on their commands at once

_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()
_pool: Optional[ThreadPoolExecutor] = None
_live: Dict[str, "Sandbox360Container"] = {}
_renewer: Optional[threading.Thread] = None


class Sandbox360Error(RuntimeError):
    """The platform refused or lost a request (not a command that exited non-zero)."""

    def __init__(self, message: str, error_code: str = "", status: int = 0):
        super().__init__(message)
        self.error_code = error_code
        self.status = status


class Sandbox360ExecResult:
    """The shape enroot and docker exec results have: ``exit_code``, ``output`` (stdout then stderr, bytes)."""

    def __init__(self, exit_code: int, output: bytes):
        self.exit_code = exit_code
        self.output = output

    def __iter__(self):
        return iter((self.exit_code, self.output))


def _http() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None or _client.is_closed:
            url, token = os.environ.get("SANDBOX360_GATEWAY_URL"), os.environ.get("SANDBOX360_TOKEN")
            if not (url and token):
                raise RuntimeError("the sandbox360 engine needs SANDBOX360_GATEWAY_URL and SANDBOX360_TOKEN")
            _client = httpx.Client(
                base_url=url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=httpx.Timeout(3600, connect=30),
                limits=httpx.Limits(max_connections=EXEC_THREADS, max_keepalive_connections=128),
            )
        return _client


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _client_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=EXEC_THREADS, thread_name_prefix="sandbox360")
        return _pool


async def _in_thread(fn, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(_executor(), lambda: fn(*args, **kwargs))


API_COUNTS = ("api_requests", "api_5xx", "api_transport_errors", "api_exec_requests", "api_exec_5xx", "api_exec_transport_errors")
_counts: Dict[str, Dict[str, int]] = {}  # per live sandbox: every HTTP attempt, so resends count too


def _count(path: str, what: str) -> None:
    parts = path.split("/")  # /v1/sandboxes/{id}/{route}
    counts = _counts.get(parts[3]) if len(parts) > 3 else None
    if counts is None:
        return
    with _client_lock:
        counts[f"api_{what}"] += 1
        if len(parts) > 4 and parts[4].startswith("exec"):
            counts[f"api_exec_{what}"] += 1


def request_counts(sandbox_id: str) -> Dict[str, int]:
    """HTTP attempts for this sandbox so far, in total and for commands. Its create call and final delete are not in it."""
    with _client_lock:
        return dict(_counts.get(sandbox_id) or dict.fromkeys(API_COUNTS, 0))


def _request(method: str, path: str, ok: Iterable[int] = (200, 202, 204), resend_on_transport: bool = True,
             **kwargs) -> httpx.Response:
    """One API call, resending what the API marks retryable (after its Retry-After). Transport errors are resent only
    for calls that are safe to repeat (``resend_on_transport``)."""
    message, code, status = "", "", 0
    for attempt in range(RETRIES):
        wait = random.uniform(0, min(30.0, 2.0**attempt))
        _count(path, "requests")
        try:
            reply = _http().request(method, path, **kwargs)
        except httpx.HTTPError as e:
            _count(path, "transport_errors")
            message, code = f"{method} {path}: {type(e).__name__}: {e}", "TRANSPORT"
            print(f"[sandbox360] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {method} {path} -> transport error "
                  f"{type(e).__name__}: {str(e)[:300]}", flush=True)
            # A failed connect never sent the request, so it is safe to resend even an exec.
            never_sent = isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
            if not resend_on_transport and not never_sent:
                raise Sandbox360Error(message, code) from e
        else:
            if reply.status_code in ok:
                return reply
            try:
                body = reply.json()
            except ValueError:
                body = {}
            status, code = reply.status_code, str(body.get("error_code", ""))
            message = f"{method} {path}: HTTP {status} {code} {str(body.get('message', ''))[:300]}"
            if status >= 500:
                _count(path, "5xx")
                # Enough to tell a load balancer's reply from the service's, and to trace it on their side.
                # UTC time (ours and the replying server's) so a report can name the minute; the sandbox id is in path.
                print(f"[sandbox360] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {method} {path} -> {status} "
                      f"code={code!r} retryable={body.get('retryable')} server={reply.headers.get('server')!r} "
                      f"date={reply.headers.get('date')!r} trace={reply.headers.get('x-amzn-trace-id')!r} "
                      f"body={reply.text[:1500]!r}", flush=True)
            # A load balancer's bare 502/503/504: resend only what is safe to repeat (API docs, section 12).
            upstream = status in (502, 503, 504) and code in ("", "UPSTREAM_UNAVAILABLE")
            if not (body.get("retryable") or (resend_on_transport and upstream)):
                raise Sandbox360Error(message, code, status)
            wait += float(reply.headers.get("Retry-After", 0) or 0)
        time.sleep(wait)
    raise Sandbox360Error(message, code, status)


def _renew_loop() -> None:
    while True:
        time.sleep(RENEW_EVERY)
        for box in list(_live.values()):
            try:
                _request("POST", f"/v1/sandboxes/{box.sandbox_id}/renew")
            except Exception as e:  # noqa: BLE001  (the next exec reports a dead sandbox)
                logger.warning("[sandbox360] renew failed for %s: %s", box.sandbox_id, e)


def _track(box: "Sandbox360Container") -> None:
    global _renewer
    with _client_lock:
        _live[box.name] = box
        _counts[box.sandbox_id] = dict.fromkeys(API_COUNTS, 0)
        if _renewer is None or not _renewer.is_alive():
            _renewer = threading.Thread(target=_renew_loop, name="sandbox360-renew", daemon=True)
            _renewer.start()


class Sandbox360Container(ContainerResource):
    """A remote sandbox that is both the resource and its own handle (as ``SqsContainer``)."""

    def __init__(self, name: str, sandbox_id: str, image: Optional[str] = None, workdir: Optional[str] = None,
                 spec: Optional[Any] = None):
        self.name = name
        self.sandbox_id = sandbox_id
        self.image = image
        self.workdir = workdir
        self._status = "running"
        super().__init__(container=self, resource_id=name, spec=spec)

    # -- lifecycle --
    @property
    def status(self) -> str:
        return self._status

    def reload(self) -> None:
        if self._status == "exited":
            return
        try:
            state = _request("GET", f"/v1/sandboxes/{self.sandbox_id}").json().get("status")
            self._status = "running" if state == "ready" else "exited"
        except Sandbox360Error:
            self._status = "exited"

    async def is_alive(self) -> bool:
        """Whether the sandbox still answers as ready (after a lost command, for example)."""
        await self.reload_async()
        return self._status == "running"

    async def reload_async(self) -> None:
        await _in_thread(self.reload)

    def kill(self) -> None:
        if self._status == "exited" and self.name not in _live:
            return
        self._status = "exited"
        _live.pop(self.name, None)
        with _client_lock:
            _counts.pop(self.sandbox_id, None)
        try:
            _request("DELETE", f"/v1/sandboxes/{self.sandbox_id}")
        except Sandbox360Error as e:
            logger.warning("[sandbox360] delete failed for %s (its lease expires): %s", self.sandbox_id, e)

    async def kill_async(self, timeout: int = 10, **_) -> None:
        await _in_thread(self.kill)

    # -- exec --
    def exec_run(self, cmd, workdir=None, user=None, environment=None, timeout=None, **_) -> Sandbox360ExecResult:
        argv = ["sh", "-c", cmd] if isinstance(cmd, str) else [str(part) for part in cmd]
        body: Dict[str, Any] = {"Cmd": argv, "Container": "task"}
        if workdir or self.workdir:
            body["WorkingDir"] = str(workdir or self.workdir)
        if environment:
            body["Env"] = {str(k): str(v) for k, v in environment.items()}
        if user:
            body["User"] = user
        if timeout:
            body["Timeout"] = float(timeout)
        try:
            # Without a read deadline a lost reply would hold the episode for the client's whole timeout (one hour).
            read = httpx.Timeout(float(timeout) + 60, connect=30) if timeout else httpx.USE_CLIENT_DEFAULT
            data = _request("POST", f"/v1/sandboxes/{self.sandbox_id}/exec_run", ok=(200,),
                            resend_on_transport=False, json=body, timeout=read).json()
        except Sandbox360Error as e:
            if e.error_code == "EXEC_TIMEOUT" or "ReadTimeout" in str(e):
                raise asyncio.TimeoutError(str(e)) from e
            raise
        if "ExitCode" not in data:
            raise Sandbox360Error(f"exec without an exit code: {str(data)[:300]}", str(data.get("error_code", "")))
        output = base64.b64decode(data.get("Stdout") or "") + base64.b64decode(data.get("Stderr") or "")
        return Sandbox360ExecResult(int(data["ExitCode"]), output)

    async def exec_run_async(self, cmd, workdir=None, user=None, environment=None, timeout=None,
                             **_) -> Sandbox360ExecResult:
        return await _in_thread(self.exec_run, cmd, workdir, user, environment, timeout)

    # -- files --
    def put_archive(self, path: str, data: Union[bytes, Any], timeout: Optional[float] = None) -> bool:
        blob = data if isinstance(data, bytes) else bytes(data)
        _request("PUT", f"/v1/sandboxes/{self.sandbox_id}/archive", params={"path": path or "/", "Container": "task"},
                 content=blob, headers={"Content-Type": "application/x-tar"}, resend_on_transport=False)
        return True

    async def put_archive_async(self, path: str, data: bytes, timeout: Optional[float] = None) -> bool:
        return await _in_thread(self.put_archive, path, data, timeout)

    def get_archive_bytes(self, path: str) -> bytes:
        return _request("GET", f"/v1/sandboxes/{self.sandbox_id}/archive",
                        params={"path": path, "Container": "task"}).content

    async def get_archive_async(self, path: str) -> bytes:
        return await _in_thread(self.get_archive_bytes, path)

    def copy_to(self, local_path: str, container_path: str, timeout: Optional[float] = None) -> bool:
        source = Path(local_path)
        if not source.exists():
            logger.warning("[sandbox360] copy_to: %s does not exist", local_path)
            return False
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            tar.add(str(source), arcname=os.path.basename(container_path.rstrip("/")))
        return self.put_archive(os.path.dirname(container_path.rstrip("/")) or "/", buffer.getvalue(), timeout)

    async def copy_to_async(self, local_path: str, container_path: str, timeout: Optional[float] = None) -> bool:
        return await _in_thread(self.copy_to, local_path, container_path, timeout)

    def copy_from(self, container_path: str, local_dir: str, timeout: Optional[float] = None) -> bool:
        data = self.get_archive_bytes(container_path)
        if not data:
            return False
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(local_dir, filter="data")
        return True


def start_sandbox360_container(image: Optional[str], name: str, environment: Optional[Dict[str, str]] = None,
                               workdir: Optional[str] = None, timeout: float = 1800.0,
                               spec: Optional[Any] = None) -> Sandbox360Container:
    """Create a sandbox, wait until it is ready, and keep its lease renewed."""
    if not image:
        raise ValueError("the sandbox360 engine needs spec.image")
    container: Dict[str, Any] = {"Name": "task", "Image": image}
    if environment:
        container["Env"] = {str(k): str(v) for k, v in environment.items()}
    body: Dict[str, Any] = {"IdempotencyKey": str(uuid.uuid4()), "LeaseSeconds": LEASE_SECONDS,
                            "Requires": ["subprocess"], "ExecEnv": True, "Containers": [container]}
    if spec is not None and getattr(spec, "mem_limit", None):
        body["MemoryLimit"] = str(spec.mem_limit)
    sandbox_id = _request("POST", "/v1/sandboxes/create", json=body).json()["sandbox_id"]
    box = Sandbox360Container(name, sandbox_id, image=image, workdir=workdir, spec=spec)
    _track(box)
    deadline = time.monotonic() + timeout
    while True:
        state = _request("GET", f"/v1/sandboxes/{sandbox_id}").json()
        status = state.get("status")
        if status == "ready":
            logger.info("[sandbox360] ready: %s (%s)", sandbox_id[:12], name)
            return box
        if status != "provisioning" or time.monotonic() > deadline:
            box.kill()
            raise Sandbox360Error(
                f"sandbox {sandbox_id} not ready: {status} {state.get('error_code', '')} {str(state.get('reason', ''))[:300]}",
                str(state.get("error_code", "")),
            )
        time.sleep(READY_POLL)
