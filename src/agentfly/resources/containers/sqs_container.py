"""SQS/EKS container backend — sandboxes as Kubernetes pods, driven over SQS.

Every container operation (create / exec / archive / delete) is serialised into
an SQS message and handled by ``docker_k8s_consumer`` running on EKS; images are
built server-side with Kaniko into ECR, and large payloads travel via S3. There
is no local daemon and no per-org memory quota, which is what makes this the
scalable alternative to the ``daytona`` engine.

Wire protocol (mirrors the harbor-side client this was ported from):

    message  {req_id, channel, method, path, query, headers, content,
              compress, sent_at, task_id, **extra}
    paths    POST v1.43/containers/create
             POST v1.43/exec_run/<cid>
             PUT  v1.43/containers/<cid>/archive?path=<dir>
             GET  v1.43/containers/<cid>/archive?path=<p>
             DELETE v1.43/containers/<cid>?force=true
    replies  {"type": "FULL"|"STREAM"|"STREAM_END", status_code, content,
              compress, content_type, exit_code, error_code}

Config comes from the resource spec (``sqs_queue_url``/``s3_bucket``/
``registry_url``/regions) or the matching ``AF_SQS_*`` env vars. AWS creds come
from the boto3 chain.

Selected with ``container_engine="sqs"`` (or ``AF_CONTAINER_ENGINE=sqs``).
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import io
import json
import logging
import os
import random
import re
import shlex
import tarfile
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from .container_resource import ContainerResource

logger = logging.getLogger(__name__)

_API = "v1.43"
_S3_THRESHOLD = 200 * 1024
_BUILD_TAG_VERSION = "v2"
_DEFAULT_BUILD_REPOSITORY = "8c8cae2f-3503-468a-be37-5d852496b289"
_CREATE_MAX_ATTEMPTS = 8
_CREATE_BACKOFF_BASE = 10.0
_CREATE_BACKOFF_MAX = 120.0


def _cfg(spec: Any, name: str, env: str, default: Optional[str] = None) -> Optional[str]:
    val = getattr(spec, name, None) if spec is not None else None
    if not val:
        val = os.environ.get(env)
    return val or default


def _normalise_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    return info


def _build_context_archive(dockerfile: str, context_dir: Optional[str]) -> bytes:
    """Create a deterministic context with the requested Dockerfile."""
    tar_buffer = io.BytesIO()
    root = Path(context_dir) if context_dir else None
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as tf:
        dockerfile_info = tarfile.TarInfo("Dockerfile")
        dockerfile_bytes = dockerfile.encode()
        dockerfile_info.size = len(dockerfile_bytes)
        dockerfile_info.mode = 0o644
        _normalise_tarinfo(dockerfile_info)
        tf.addfile(dockerfile_info, io.BytesIO(dockerfile_bytes))

        if root and root.is_dir():
            for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
                relative = path.relative_to(root).as_posix()
                if relative == "Dockerfile":
                    continue
                info = _normalise_tarinfo(tf.gettarinfo(str(path), arcname=relative))
                if info.isfile():
                    with path.open("rb") as source:
                        tf.addfile(info, source)
                else:
                    tf.addfile(info)

    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as gz:
        gz.write(tar_buffer.getvalue())
    return compressed.getvalue()


def _build_artifact(
    dockerfile: str,
    context_dir: Optional[str],
    repository: Optional[str] = None,
) -> tuple[str, str, bytes]:
    """Return a shared-repository build tag, deterministic S3 key, and context.

    Exact task context (the requested Dockerfile matches the one on disk in
    ``context_dir``) uses the shared ``harbor-v3`` content identity so a
    prebuilt image is reused — this is what used to be the
    ``install_agentfly_sqs_build_cache_patch`` monkey-patch, now native. A
    transformed / synthesized Dockerfile (differs from the on-disk one) uses the
    native scheme below.
    """
    build_repository = (
        repository
        or os.environ.get("AF_SQS_BUILD_REPOSITORY")
        or _DEFAULT_BUILD_REPOSITORY
    )
    ctx = Path(context_dir) if context_dir else None
    if (
        ctx is not None
        and ctx.is_dir()
        and (ctx / "Dockerfile").is_file()
        and (ctx / "Dockerfile").read_text() == dockerfile
    ):
        from .build_identity import build_artifact as _shared_build_artifact
        return _shared_build_artifact(ctx, build_repository)

    context = _build_context_archive(dockerfile, context_dir)
    digest = hashlib.sha256(context).hexdigest()
    if not re.fullmatch(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", build_repository):
        raise ValueError(f"invalid SQS build repository: {build_repository!r}")
    build_tag = f"{build_repository}:af-{_BUILD_TAG_VERSION}-{digest[:32]}"
    build_key = f"build-contexts/{build_tag}/{digest}.tar.gz"
    return build_tag, build_key, context


class SqsCreateError(RuntimeError):
    def __init__(self, status: int, response: bytes):
        response_text = response.decode("utf-8", errors="replace")
        try:
            payload = json.loads(response_text) or {}
        except json.JSONDecodeError:
            payload = {}
        self.status = status
        self.error_code = str(payload.get("error_code") or "UNKNOWN")
        self.retryable = payload.get("retryable") is True
        self.details = payload.get("details") or {}
        self.response = response_text
        message = str(payload.get("message") or response_text or "container create failed")
        super().__init__(
            f"[sqs] containers/create failed: status={status} "
            f"error_code={self.error_code} retryable={self.retryable} "
            f"message={message} details={json.dumps(self.details, ensure_ascii=False)}"
        )


class SqsExecResult:
    """docker-py-shaped exec result (``exit_code`` + ``output`` bytes)."""

    __slots__ = ("exit_code", "output", "error_code")

    def __init__(self, exit_code: int, output: bytes, error_code: Optional[str] = None):
        self.exit_code = exit_code
        self.output = output
        self.error_code = error_code

    def __iter__(self):        # allows `code, out = result`
        return iter((self.exit_code, self.output))


class SqsTransport:
    """Request/response over the docker-requests queue.

    One response queue per container: created on first use, deleted with the
    container. Replies are correlated by ``req_id`` and STREAM chunks are
    concatenated until STREAM_END.
    """

    def __init__(self, queue_url: str, region: str, s3_bucket: str,
                 s3_region: Optional[str] = None, task_id: str = "agentfly"):
        import boto3
        from botocore.config import Config as BotoConfig

        self.queue_url = queue_url
        self.s3_bucket = s3_bucket
        self.task_id = task_id
        cfg = BotoConfig(max_pool_connections=64,
                         retries={"max_attempts": 5, "mode": "standard"})
        self._sqs = boto3.client("sqs", region_name=region, config=cfg)
        self._s3 = boto3.client("s3", region_name=s3_region or region)
        self._channel: Optional[str] = None
        self.slurm_user = (os.environ.get("SLURM_JOB_USER")
                           or os.environ.get("USER") or "UNKNOWN_USER")
        self.slurm_job_id = os.environ.get("SLURM_JOB_ID") or "UNKNOWN_JOB_ID"

    # -- channel ---------------------------------------------------------
    @property
    def channel(self) -> str:
        if self._channel is None:
            name = f"af-{self.task_id[:40]}-{uuid4().hex[:8]}"
            name = re.sub(r"[^A-Za-z0-9_-]", "-", name)[:78]
            self._channel = self._sqs.create_queue(
                QueueName=name,
                Attributes={"MessageRetentionPeriod": "3600"})["QueueUrl"]
        return self._channel

    def close(self) -> None:
        if self._channel:
            try:
                self._sqs.delete_queue(QueueUrl=self._channel)
            except Exception:  # noqa: BLE001
                pass
            self._channel = None

    # -- payload codecs --------------------------------------------------
    def _encode(self, body: bytes) -> Tuple[str, bool, Optional[str]]:
        """(content, compress, s3_key) — big bodies go to S3."""
        if not body:
            return "", False, None
        if len(body) > _S3_THRESHOLD:
            key = f"agentfly-sqs/{uuid4().hex}"
            self._s3.put_object(Bucket=self.s3_bucket, Key=key, Body=body)
            return "", False, key
        return base64.b64encode(zlib.compress(body)).decode(), True, None

    def _decode(self, msg: dict) -> bytes:
        if msg.get("s3_key"):
            obj = self._s3.get_object(Bucket=self.s3_bucket, Key=msg["s3_key"])
            return obj["Body"].read()
        content = msg.get("content", "") or ""
        if msg.get("compress"):
            return zlib.decompress(base64.b64decode(content))
        ctype = msg.get("content_type", "") or ""
        if ctype.startswith(("application/x-tar", "application/octet-stream")):
            try:
                return base64.b64decode(content)
            except Exception:  # noqa: BLE001
                pass
        return content.encode() if isinstance(content, str) else (content or b"")

    # -- request/response ------------------------------------------------
    def request(self, method: str, path: str, *, query: str = "",
                headers: Optional[dict] = None, body: bytes = b"",
                extra: Optional[dict] = None,
                timeout: float = 600.0) -> Tuple[int, bytes, dict]:
        """Send one request and assemble its reply. Returns (status, body, last_msg)."""
        content, compress, s3_key = self._encode(body)
        msg = {"req_id": uuid4().hex, "channel": self.channel, "method": method,
               "path": path, "query": query, "headers": headers or {},
               "content": content, "compress": compress,
               "sent_at": time.time(), "task_id": self.task_id,
               "slurm_user": self.slurm_user, "slurm_job_id": self.slurm_job_id}
        if s3_key:
            msg["s3_key"] = s3_key
        if extra:
            msg.update(extra)
        req_id = msg["req_id"]
        self._sqs.send_message(QueueUrl=self.queue_url, MessageBody=json.dumps(msg))

        parts: List[bytes] = []
        status, last = 0, {}
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._sqs.receive_message(
                QueueUrl=self.channel, MaxNumberOfMessages=10, WaitTimeSeconds=10)
            for m in resp.get("Messages", []):
                self._sqs.delete_message(QueueUrl=self.channel,
                                         ReceiptHandle=m["ReceiptHandle"])
                try:
                    r = json.loads(m["Body"])
                except json.JSONDecodeError:
                    continue
                if r.get("req_id") != req_id:
                    continue          # stale reply from an earlier request
                last = r
                status = r.get("status_code") or status
                kind = r.get("type")
                if kind == "STREAM":
                    chunk = self._decode(r)
                    if chunk:
                        parts.append(chunk)
                    deadline = time.time() + timeout   # keepalive resets clock
                    continue
                if kind == "STREAM_END":
                    # STREAM_END's content is the consumer's end marker
                    # ("[EXIT]"), not output — the real bytes came in the
                    # STREAM chunks. Appending it corrupts every exec result.
                    chunk = self._decode(r)
                    if chunk and chunk.strip() not in (b"[EXIT]", b""):
                        parts.append(chunk)
                    return status, b"".join(parts), r
                return status, self._decode(r), r      # FULL
        raise TimeoutError(f"[sqs] no reply for {method} {path} in {timeout}s")

    # -- docker-ish operations -------------------------------------------
    def upload_build_context(self, tar_bytes: bytes, key: str) -> None:
        self._s3.put_object(Bucket=self.s3_bucket, Key=key, Body=tar_bytes)

    def create(self, image: str, cmd: List[str], env: Dict[str, str],
               binds: List[str], *, container_id: str,
               build_context_s3_key: Optional[str] = None,
               timeout_sec: int = 1800,
               memory_limit: Optional[str] = None,
               cpu_limit: Optional[float] = None) -> str:
        body: dict = {"Image": image, "Cmd": cmd, "ContainerId": container_id,
                      "AttachStdout": False, "AttachStderr": False,
                      "keepalive": True}
        if env:
            body["Env"] = [f"{k}={v}" for k, v in env.items()]
        if binds:
            body["HostConfig"] = {"Binds": binds}
        extra: dict = {"timeout_sec": timeout_sec}
        if build_context_s3_key:
            # All THREE are required for the consumer to build instead of
            # pull: without build_tag it has no build instruction and falls
            # through to an image pull -> IMAGE_PULL_FAILED.
            extra["build_tag"] = image
            extra["build_context_s3_key"] = build_context_s3_key
            extra["force_build"] = False
        if memory_limit:
            extra["memory_limit"] = memory_limit
        if cpu_limit:
            extra["cpu_limit"] = cpu_limit
        status, resp, msg = self.request(
            "POST", f"{_API}/containers/create",
            body=json.dumps(body).encode(), extra=extra,
            timeout=float(timeout_sec))
        if status >= 400:
            error = SqsCreateError(status, resp)
            logger.error("%s; response=%s", error, error.response)
            raise error
        return container_id

    def exec(self, cid: str, cmd: Union[str, List[str]], *,
             user: str = "", env: Optional[Dict[str, str]] = None,
             timeout_sec: Optional[float] = None) -> SqsExecResult:
        if isinstance(cmd, str):
            cmd = ["bash", "-c", cmd]
        body: dict = {"Cmd": cmd, "AttachStdout": True, "AttachStderr": True,
                      "Detach": False, "User": user or "",
                      "client": "sqs_env", "exec_id": uuid4().hex,
                      "keepalive": True}
        if env:
            body["Env"] = [f"{k}={v}" for k, v in env.items()]
        if timeout_sec and timeout_sec > 0:
            body["timeout_sec"] = int(timeout_sec)
        status, out, last = self.request(
            "POST", f"{_API}/exec_run/{cid}", body=json.dumps(body).encode(),
            timeout=float(timeout_sec or 600))
        exit_code = last.get("exit_code")
        if exit_code is None:
            exit_code = 0 if status < 400 else 1
        return SqsExecResult(int(exit_code), out, last.get("error_code"))

    def put_archive(self, cid: str, target_dir: str, tar_bytes: bytes,
                    timeout: float = 600.0) -> bool:
        status, _, _ = self.request(
            "PUT", f"{_API}/containers/{cid}/archive",
            query=f"path={target_dir}",
            headers={"Content-Type": "application/x-tar"},
            body=tar_bytes, timeout=timeout)
        return status < 400

    def get_archive(self, cid: str, path: str, timeout: float = 600.0) -> bytes:
        _, data, _ = self.request(
            "GET", f"{_API}/containers/{cid}/archive",
            query=f"path={path}", timeout=timeout)
        return data

    def delete(self, cid: str, timeout: float = 120.0) -> None:
        try:
            self.request("DELETE", f"{_API}/containers/{cid}",
                         query="force=true", timeout=timeout)
        except Exception as e:  # noqa: BLE001 — delete is best-effort
            logger.warning("[sqs] delete %s failed: %s", cid[:12], e)


def _make_tar(source: Path, arcname: Optional[str] = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        tf.add(str(source), arcname=arcname or source.name)
    return buf.getvalue()


class SqsContainer(ContainerResource):
    """An EKS pod that is both the resource and its own handle.

    Mirrors :class:`DaytonaContainer` / :class:`DockerContainer`: subclasses
    :class:`ContainerResource` and passes ``container=self``, so the inherited
    lifecycle and ``run_cmd`` drive the handle methods below.
    """

    def __init__(self, name: str, cid: str, transport: SqsTransport,
                 image: Optional[str] = None, workdir: Optional[str] = None,
                 spec: Optional[Any] = None):
        self.name = name
        self.cid = cid
        self.transport = transport
        self.image = image
        self.workdir = workdir
        self._status = "running"
        super().__init__(container=self, resource_id=name, spec=spec)

    # -- identity / lifecycle --------------------------------------------
    @property
    def sandbox_id(self) -> Optional[str]:
        return self.cid

    @property
    def status(self) -> str:
        return self._status

    def reload(self) -> None:
        """Cheap liveness check — an exec that fails means the pod is gone."""
        try:
            res = self.transport.exec(self.cid, "true", timeout_sec=30)
            self._status = "running" if res.exit_code == 0 else "exited"
        except Exception:  # noqa: BLE001
            self._status = "exited"

    async def reload_async(self) -> None:
        await asyncio.to_thread(self.reload)

    def kill(self) -> None:
        self.transport.delete(self.cid)
        self._status = "exited"
        self.transport.close()

    async def kill_async(self, timeout: int = 10, **_) -> None:
        await asyncio.to_thread(self.kill)

    def preview_link(self, port: int) -> Optional[str]:
        return None            # no ingress for sandbox pods

    # -- exec -------------------------------------------------------------
    def _exec(self, cmd, workdir=None, user=None, environment=None,
              timeout=None) -> SqsExecResult:
        wd = workdir or self.workdir
        if wd:
            # shlex.join (NOT " ".join): run_cmd passes argv ["bash","-c",<script>];
            # a plain-space join drops the quoting so `bash -c` sees only the
            # first word of the script and the rest leak to the outer shell —
            # every multi-word container command silently runs only its first
            # token. Quote each argv element (and the workdir) so the script
            # stays one argument.
            inner = cmd if isinstance(cmd, str) else shlex.join(str(p) for p in cmd)
            cmd = f"cd {shlex.quote(str(wd))} && {inner}"
        return self.transport.exec(
            self.cid, cmd, user=str(user) if user else "",
            env=environment or None, timeout_sec=timeout)

    def exec_run(self, cmd, workdir=None, user=None, environment=None,
                 timeout=None, **_) -> SqsExecResult:
        return self._exec(cmd, workdir, user, environment, timeout)

    async def exec_run_async(self, cmd, workdir=None, user=None,
                             environment=None, timeout=None, **_) -> SqsExecResult:
        return await asyncio.to_thread(
            self._exec, cmd, workdir, user, environment, timeout)

    # -- copy -------------------------------------------------------------
    def put_archive(self, path: str, data: Union[bytes, Any],
                    timeout: Optional[float] = None) -> bool:
        blob = data if isinstance(data, bytes) else bytes(data)
        return self.transport.put_archive(self.cid, path or "/", blob,
                                          timeout=timeout or 600.0)

    def copy_to(self, local_path: str, container_path: str,
                timeout: Optional[float] = None) -> bool:
        src = Path(local_path)
        if not src.exists():
            logger.warning("[sqs] copy_to: %s does not exist", local_path)
            return False
        target_dir = os.path.dirname(container_path.rstrip("/")) or "/"
        tar = _make_tar(src, arcname=os.path.basename(container_path.rstrip("/")))
        return self.put_archive(target_dir, tar, timeout=timeout)

    async def copy_to_async(self, local_path: str, container_path: str,
                            timeout: Optional[float] = None) -> bool:
        return await asyncio.to_thread(self.copy_to, local_path,
                                       container_path, timeout)

    def copy_from(self, container_path: str, local_dir: str,
                  timeout: Optional[float] = None) -> bool:
        data = self.transport.get_archive(self.cid, container_path,
                                          timeout=timeout or 600.0)
        if not data:
            return False
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(local_dir)
        return True


def start_sqs_container(image: Optional[str], name: str,
                        environment: Optional[Dict[str, str]] = None,
                        mount: Optional[Dict[str, str]] = None,
                        ports: Optional[Any] = None,
                        workdir: Optional[str] = None,
                        timeout: float = 1800.0,
                        spec: Optional[Any] = None) -> SqsContainer:
    """Create + start one sandbox pod over SQS and return its handle.

    ``spec.dockerfile`` (+ optional ``spec.build_context``) is uploaded to S3
    so the consumer can build with Kaniko when the tag is not yet in ECR;
    otherwise ``image`` is used as-is.
    """
    queue_url = _cfg(spec, "sqs_queue_url", "AF_SQS_QUEUE_URL")
    s3_bucket = _cfg(spec, "s3_bucket", "AF_SQS_S3_BUCKET")
    region = _cfg(spec, "sqs_region", "AF_SQS_REGION", "eu-west-1")
    s3_region = _cfg(spec, "s3_region", "AF_SQS_S3_REGION", region)
    if not (queue_url and s3_bucket):
        raise RuntimeError(
            "sqs engine needs sqs_queue_url + s3_bucket (spec fields or "
            "AF_SQS_QUEUE_URL / AF_SQS_S3_BUCKET)")

    transport = SqsTransport(queue_url, region, s3_bucket, s3_region,
                             task_id=name)
    dockerfile = getattr(spec, "dockerfile", None) if spec else None
    build_key = None
    tag = image
    if dockerfile:
        ctx_dir = getattr(spec, "build_context", None)
        build_repository = _cfg(
            spec,
            "sqs_build_repository",
            "AF_SQS_BUILD_REPOSITORY",
            _DEFAULT_BUILD_REPOSITORY,
        )
        tag, build_key, ctx = _build_artifact(
            dockerfile, ctx_dir, build_repository
        )
        transport.upload_build_context(ctx, build_key)
        logger.info("[sqs] uploaded build context (%db) -> s3://%s/%s",
                    len(ctx), s3_bucket, build_key)
    if not tag:
        raise ValueError("sqs engine needs spec.image or spec.dockerfile")

    binds = [f"{host}:{cont}" for cont, host in (mount or {}).items()]
    spec_extra = getattr(spec, "extra", None) or {}
    max_attempts = int(spec_extra.get(
        "sqs_create_max_attempts",
        os.environ.get("AF_SQS_CREATE_MAX_ATTEMPTS", _CREATE_MAX_ATTEMPTS),
    ))
    cid = ""
    for attempt in range(1, max(1, max_attempts) + 1):
        cid = uuid4().hex
        try:
            transport.create(tag, ["/bin/sh", "-c", "sleep infinity"],
                             environment or {}, binds, container_id=cid,
                             build_context_s3_key=build_key,
                             timeout_sec=int(timeout),
                             memory_limit=getattr(spec, "mem_limit", None) if spec else None,
                             cpu_limit=getattr(spec, "cpu_limit", None) if spec else None)
            break
        except SqsCreateError as error:
            if not error.retryable or attempt >= max_attempts:
                transport.close()
                raise
            transport.delete(cid, timeout=30.0)
            delay = min(
                _CREATE_BACKOFF_BASE * (2 ** (attempt - 1)),
                _CREATE_BACKOFF_MAX,
            ) * random.uniform(0.5, 1.5)
            logger.warning(
                "[sqs] create retry %d/%d after %s; sleeping %.1fs",
                attempt, max_attempts, error.error_code, delay,
            )
            time.sleep(delay)
        except Exception:
            transport.close()
            raise
    logger.info("[sqs] sandbox created: %s (%s)", cid[:12], name)
    return SqsContainer(name, cid, transport, image=tag, workdir=workdir,
                        spec=spec)
