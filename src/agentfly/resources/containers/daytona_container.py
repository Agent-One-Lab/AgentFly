"""Daytona-backed container resource.

:class:`DaytonaContainer` is the Daytona analogue of :class:`DockerContainer`:
a :class:`ContainerResource` that is *its own container handle*, implementing
the enroot-``Container`` subset that :class:`ContainerResource` drives, and
passing ``container=self`` to the base so the inherited ``run_cmd`` / ``start``
/ ``get_status`` / ``end`` / ``close`` lifecycle calls back into these methods:

    - ``name``                  attribute (and ``resource_id`` == ``name``)
    - ``status``                attribute (refreshed by ``reload``)
    - ``reload()`` / ``reload_async()``
    - ``exec_run_async(cmd, workdir=, user=, environment=, timeout=)`` ->
      object with ``.exit_code`` and ``.output`` (bytes, stdout+stderr)
    - ``kill_async()``
    - ``put_archive`` / ``copy_to`` / ``copy_to_async``

Backed by a *cloud* sandbox, which differs from docker in three ways the
callers must know about:

1. **There is no local image build and no registry.** Daytona builds images
   server-side from a Dockerfile plus its COPY context. So instead of a
   pre-built image tag, a spec carries ``dockerfile`` (text) and optional
   ``build_context`` (a local dir the Dockerfile's ``COPY`` lines resolve
   against). ``image`` alone still works and is treated as an existing
   snapshot name, else a registry base image.

2. **Building per rollout would be ruinous**, so a Dockerfile+context is
   content-hashed into a named *snapshot* (``agentfly-<hash12>``) that is built
   once and reused by every later sandbox. This is the docker ``select_docker_host``
   analogue: same input -> same reusable artifact, no coordination needed.
   Registering a snapshot needs write scope on the Daytona org, which not every
   API key has; when it is denied the engine falls back — permanently, for the
   process — to a declarative per-sandbox build, which every key can do. The
   fallback is correct but slower: it rebuilds for every container instead of
   once per image.

3. **There are no bind mounts.** ``spec.mount`` is emulated by uploading each
   local source into the target path once, at start. ``ro`` is not enforced —
   a mount whose point is read-only isolation will not get it here.

Credentials come from ``DAYTONA_API_KEY`` (required); ``DAYTONA_API_URL`` and
``DAYTONA_TARGET`` are honored when set.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..types import ContainerResourceSpec, DaytonaResourceSpec
from .container_resource import ContainerResource

logger = logging.getLogger(__name__)

# Ceilings + lifecycle timeouts. Defaults live on DaytonaResourceSpec; a spec
# can carry its own via ``spec.daytona``. Resolved with ``_daytona_for(spec)``.
_DEFAULT_DAYTONA = DaytonaResourceSpec()


def _daytona_for(spec: Optional[ContainerResourceSpec]) -> DaytonaResourceSpec:
    """The spec's DaytonaResourceSpec, or the module defaults."""
    return (getattr(spec, "daytona", None) if spec else None) or _DEFAULT_DAYTONA


# Label stamped on everything we create, so a reaper can find our sandboxes
# without touching anyone else's.
_MANAGED_LABEL = {"agentfly.managed": "1"}

_IGNORE_CONTEXT = shutil.ignore_patterns(".git", "__pycache__", "*.pyc")

# Flipped once a snapshot write is refused, so the remaining containers in the
# run go straight to the declarative path instead of re-probing a denied API
# call (and paying its latency) on every single acquire. Guarded by a lock: the
# check-and-set is not atomic, so without it every container in the first
# concurrent batch probes (and logs) the same denial.
_snapshots_denied = False
_snapshots_denied_lock = threading.Lock()


class SnapshotUnavailable(RuntimeError):
    """Snapshot registration is not permitted for these credentials.

    Distinct from a build FAILURE: a bad Dockerfile must surface as an error,
    while a permissions gap should transparently degrade to a declarative build.
    """


@dataclass
class DaytonaExecResult:
    """docker-py / enroot ExecResult-compatible: (exit_code, output)."""
    exit_code: int
    output: bytes

    def __iter__(self):
        return iter((self.exit_code, self.output))

    def __getitem__(self, i):
        return (self.exit_code, self.output)[i]


# --- client -----------------------------------------------------------------

_client_lock = threading.Lock()
_client: Any = None


def daytona_client() -> Any:
    """Process-wide sync ``Daytona`` client (created once, reused).

    The SDK client owns a connection pool; building one per container would
    exhaust file descriptors under the concurrency the eval runs at.
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            from daytona import Daytona, DaytonaConfig
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The daytona engine requires the Daytona SDK: pip install daytona"
            ) from e
        if not os.environ.get("DAYTONA_API_KEY"):
            raise RuntimeError(
                "DAYTONA_API_KEY is not set — the daytona container engine "
                "cannot authenticate."
            )
        kwargs: Dict[str, Any] = {"api_key": os.environ["DAYTONA_API_KEY"]}
        if os.environ.get("DAYTONA_API_URL"):
            kwargs["api_url"] = os.environ["DAYTONA_API_URL"]
        if os.environ.get("DAYTONA_TARGET"):
            kwargs["target"] = os.environ["DAYTONA_TARGET"]
        _client = Daytona(DaytonaConfig(**kwargs))
        return _client


# --- resources --------------------------------------------------------------

def _parse_mem_gb(mem_limit: Optional[Union[str, int]]) -> Optional[int]:
    """Docker-style ``mem_limit`` -> whole GiB (rounded up), or None.

    Accepts ``"4g"``, ``"512m"``, ``"1024k"``, a byte count, or ``None``.
    Daytona's ``Resources.memory`` is an integer count of GiB, so anything
    under 1 GiB still has to ask for 1.
    """
    if mem_limit is None:
        return None
    if isinstance(mem_limit, (int, float)):
        byts = float(mem_limit)
    else:
        s = str(mem_limit).strip().lower()
        mult = 1.0
        if s.endswith("g") or s.endswith("gb"):
            mult, s = 1024 ** 3, s.rstrip("b").rstrip("g")
        elif s.endswith("m") or s.endswith("mb"):
            mult, s = 1024 ** 2, s.rstrip("b").rstrip("m")
        elif s.endswith("k") or s.endswith("kb"):
            mult, s = 1024, s.rstrip("b").rstrip("k")
        try:
            byts = float(s) * mult
        except ValueError:
            return None
    return max(1, int(-(-byts // (1024 ** 3))))


def _resources_for(spec: Optional[ContainerResourceSpec]) -> Any:
    """Build a clamped ``Resources`` from the spec, or None to take defaults."""
    from daytona import Resources

    tuning = _daytona_for(spec)
    kwargs: Dict[str, Any] = {}
    cpus = getattr(spec, "cpu_count", None) if spec else None
    if cpus:
        want = max(1, int(cpus))
        if want > tuning.max_cpus:
            logger.warning("daytona: clamping cpus %s -> %s", want, tuning.max_cpus)
        kwargs["cpu"] = min(want, tuning.max_cpus)
    mem_gb = _parse_mem_gb(getattr(spec, "mem_limit", None) if spec else None)
    if mem_gb:
        if mem_gb > tuning.max_memory_gb:
            logger.warning("daytona: clamping memory %sGiB -> %sGiB",
                           mem_gb, tuning.max_memory_gb)
        kwargs["memory"] = min(mem_gb, tuning.max_memory_gb)
    disk_gb = getattr(spec, "disk_gb", None) if spec else None
    if disk_gb:
        if int(disk_gb) > tuning.max_disk_gb:
            logger.warning("daytona: clamping disk %sGiB -> %sGiB",
                           int(disk_gb), tuning.max_disk_gb)
        kwargs["disk"] = min(int(disk_gb), tuning.max_disk_gb)
    gpus = getattr(spec, "gpus", None) if spec else None
    if gpus:
        try:
            n = int(gpus)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            kwargs["gpu"] = n
    return Resources(**kwargs) if kwargs else None


# --- image / snapshot -------------------------------------------------------

def _context_digest(dockerfile: str, build_context: Optional[str]) -> str:
    """Stable hash of the Dockerfile plus every file it could COPY in.

    Two runs with byte-identical inputs must produce the same snapshot name so
    the second one reuses the first one's build; any content change must produce
    a different name so a stale image is never silently reused.
    """
    h = hashlib.sha256()
    h.update(dockerfile.encode("utf-8"))
    if build_context and os.path.isdir(build_context):
        root = Path(build_context)
        for p in sorted(root.rglob("*")):
            if p.is_dir() or not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if rel.startswith(".git/") or "__pycache__" in rel:
                continue
            h.update(rel.encode("utf-8"))
            try:
                h.update(p.read_bytes())
            except OSError:
                h.update(b"<unreadable>")
    return h.hexdigest()[:12]


def _snapshot_name(digest: str) -> str:
    """``agentfly-<digest>``, suffixed by target when one is pinned.

    Snapshots are per-target, so the same content built against two targets
    must not collide on one name.
    """
    target = os.environ.get("DAYTONA_TARGET", "").strip().lower()
    base = f"agentfly-{digest}"
    return f"{base}-{target}" if target else base


def _image_from(dockerfile: str, build_context: Optional[str]) -> Any:
    """A Daytona ``Image`` for a Dockerfile + its COPY context.

    The SDK resolves ``COPY`` sources relative to the Dockerfile's own
    directory, so the context is materialized into a temp dir with the
    Dockerfile written alongside it. The temp dir must outlive this call only
    until the snapshot build uploads it, which ``snapshot.create`` does
    synchronously — hence the explicit cleanup by the caller.
    """
    from daytona import Image

    tmp = tempfile.mkdtemp(prefix="agentfly-daytona-ctx-")
    if build_context and os.path.isdir(build_context):
        shutil.copytree(build_context, tmp, dirs_exist_ok=True, symlinks=True,
                        ignore=_IGNORE_CONTEXT, ignore_dangling_symlinks=True)
    df_path = os.path.join(tmp, "Dockerfile")
    with open(df_path, "w", encoding="utf-8") as f:
        f.write(dockerfile if dockerfile.endswith("\n") else dockerfile + "\n")
    return Image.from_dockerfile(df_path), tmp


# In-process serialization: without it, N concurrent rollouts sharing one
# Dockerfile would each start the same build. Cross-process races are handled
# by treating a create conflict as "someone else is building it" and waiting.
_snapshot_locks: Dict[str, threading.Lock] = {}
_snapshot_locks_guard = threading.Lock()


def _lock_for(name: str) -> threading.Lock:
    with _snapshot_locks_guard:
        return _snapshot_locks.setdefault(name, threading.Lock())


def _snapshot_state(client: Any, name: str) -> Optional[str]:
    """Current snapshot state as a lowercase string, or None if absent."""
    try:
        snap = client.snapshot.get(name)
    except Exception:  # noqa: BLE001 - SDK raises NotFound and transport errors alike
        return None
    state = getattr(snap, "state", None)
    return str(getattr(state, "value", state) or "").lower() or None


def _wait_active(client: Any, name: str, timeout: float) -> bool:
    """Poll until the snapshot is ACTIVE. False on error state or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _snapshot_state(client, name)
        if state == "active":
            return True
        if state in ("error", "build_failed"):
            return False
        time.sleep(5)
    return False


def ensure_snapshot(dockerfile: str, build_context: Optional[str] = None,
                    *, resources: Any = None,
                    timeout: float = _DEFAULT_DAYTONA.snapshot_build_timeout) -> str:
    """Build-once/reuse a snapshot for ``dockerfile`` and return its name.

    Idempotent and safe to call concurrently from many rollouts: an existing
    ACTIVE snapshot short-circuits, a build in flight is waited on, and a
    previously FAILED snapshot is deleted and rebuilt (a permanently poisoned
    name would otherwise fail every future run).
    """
    client = daytona_client()
    name = _snapshot_name(_context_digest(dockerfile, build_context))

    if _snapshot_state(client, name) == "active":
        return name

    with _lock_for(name):
        state = _snapshot_state(client, name)
        if state == "active":
            return name
        if state in ("building", "pending", "pulling"):
            if _wait_active(client, name, timeout):
                return name
            raise RuntimeError(f"daytona snapshot {name} did not become active")
        if state in ("error", "build_failed"):
            logger.warning("daytona: rebuilding failed snapshot %s", name)
            try:
                client.snapshot.delete(client.snapshot.get(name))
            except Exception:  # noqa: BLE001
                pass

        from daytona import CreateSnapshotParams

        image, ctx_dir = _image_from(dockerfile, build_context)
        try:
            client.snapshot.create(
                CreateSnapshotParams(name=name, image=image, resources=resources),
                on_logs=lambda chunk: logger.debug("daytona build %s: %s",
                                                   name, chunk.rstrip()),
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            # A conflict means another process won the race — wait it out
            # instead of failing this rollout.
            if "already exists" in msg or "conflict" in msg:
                if _wait_active(client, name, timeout):
                    return name
            if ("access denied" in msg or "forbidden" in msg
                    or "unauthorized" in msg or "permission" in msg):
                raise SnapshotUnavailable(str(e)) from e
            raise RuntimeError(f"daytona snapshot build failed for {name}: {e}") from e
        finally:
            shutil.rmtree(ctx_dir, ignore_errors=True)

        if not _wait_active(client, name, timeout):
            raise RuntimeError(f"daytona snapshot {name} did not become active")
        return name


# --- start ------------------------------------------------------------------

def _delete_by_name_label(client: Any, cname: str) -> int:
    """Delete sandboxes tagged with our ``agentfly.name`` label. Best-effort."""
    try:
        from daytona import ListSandboxesQuery
        stale = list(client.list(ListSandboxesQuery(
            labels={**_MANAGED_LABEL, "agentfly.name": cname})))
    except Exception as e:  # noqa: BLE001
        logger.debug("daytona: could not list sandboxes for %s: %s", cname, e)
        return 0
    n = 0
    for sb in stale:
        try:
            sb.delete()
            n += 1
        except Exception:  # noqa: BLE001
            pass
    if n:
        logger.info("daytona: cleaned up %d failed sandbox(es) for %s", n, cname)
    return n


def start_daytona_container(
    image: Optional[str] = None,
    name: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    mount: Optional[Dict[str, str]] = None,
    ports: Optional[Dict[str, Any]] = None,
    workdir: Optional[str] = None,
    timeout: Optional[float] = None,
    spec: Optional[ContainerResourceSpec] = None,
) -> "DaytonaContainer":
    """Create a Daytona sandbox and return a handle.

    Image resolution, in precedence order:

    1. ``spec.snapshot`` — an existing snapshot name, used as-is.
    2. ``spec.dockerfile`` (+ ``spec.build_context``) — content-hashed into a
       snapshot that is built once and reused (see :func:`ensure_snapshot`).
    3. ``image`` — an existing snapshot of that name if there is one, else a
       registry image pulled as the sandbox base.

    Unlike docker there is no ``sleep infinity`` keep-alive: a Daytona sandbox
    is a persistent VM, so ``exec`` has somewhere to run from the moment it is
    created.
    """
    from daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams

    client = daytona_client()
    tuning = _daytona_for(spec)
    cname = name or f"agentfly-{uuid.uuid4().hex[:12]}"
    resources = _resources_for(spec)
    create_timeout = tuning.create_timeout if timeout is None else float(timeout)

    common: Dict[str, Any] = {
        "env_vars": dict(environment or {}),
        "labels": {**_MANAGED_LABEL, "agentfly.name": cname},
        "auto_stop_interval": tuning.auto_stop_min,
        "auto_delete_interval": tuning.auto_delete_min,
    }

    global _snapshots_denied
    snapshot = getattr(spec, "snapshot", None) if spec else None
    dockerfile = getattr(spec, "dockerfile", None) if spec else None
    build_context = getattr(spec, "build_context", None) if spec else None

    with _snapshots_denied_lock:
        try_snapshot = (dockerfile and tuning.snapshots != "off"
                        and not _snapshots_denied)
    if not snapshot and try_snapshot:
        try:
            snapshot = ensure_snapshot(dockerfile, build_context, resources=resources,
                                       timeout=tuning.snapshot_build_timeout)
        except SnapshotUnavailable as e:
            if tuning.snapshots == "on":
                raise
            with _snapshots_denied_lock:
                first = not _snapshots_denied
                _snapshots_denied = True
            if first:
                logger.warning(
                    "daytona: snapshot registration denied (%s) — falling back to a "
                    "declarative per-sandbox build for the rest of this run. Images "
                    "will be rebuilt per container; grant snapshot write scope to "
                    "this API key to get build-once reuse.", e)
    if not snapshot and image and _snapshot_state(client, image) == "active":
        snapshot = image

    ctx_dir = None
    if snapshot:
        params = CreateSandboxFromSnapshotParams(snapshot=snapshot, **common)
    elif dockerfile:
        # Declarative build: the Dockerfile + context are shipped with the
        # create call and built server-side, per sandbox.
        built, ctx_dir = _image_from(dockerfile, build_context)
        params = CreateSandboxFromImageParams(
            image=built, resources=resources, **common)
    else:
        if not image:
            raise ValueError(
                "daytona engine needs one of spec.snapshot, spec.dockerfile, "
                "or image"
            )
        params = CreateSandboxFromImageParams(
            image=image, resources=resources, **common)

    try:
        sandbox = client.create(
            params,
            timeout=create_timeout,
            on_snapshot_create_logs=lambda chunk: logger.debug(
                "daytona build %s: %s", cname, chunk.rstrip()),
        )
    except BaseException:
        # A create that fails DURING the build still leaves a sandbox record
        # behind (state BUILD_FAILED), and it never reaches us as a handle — so
        # a retry loop silently accumulates them. Find ours by label and remove
        # them here, since nothing else holds a reference.
        _delete_by_name_label(client, cname)
        raise
    finally:
        # The context dir must outlive create() — it is uploaded during the
        # call — but not a moment longer.
        if ctx_dir:
            shutil.rmtree(ctx_dir, ignore_errors=True)

    c = DaytonaContainer(cname, sandbox=sandbox, image=snapshot or image,
                         workdir=workdir, spec=spec)
    try:
        c.reload()
        if workdir:
            c._exec(f"mkdir -p {shlex.quote(workdir)}", None, None, None, 60)
        # No bind mounts on a cloud sandbox: materialize each mount source.
        for src, dest_spec in (mount or {}).items():
            dest = dest_spec.split(":", 1)[0]
            if not os.path.exists(src):
                continue
            if ":" in dest_spec and "ro" in dest_spec.split(":", 1)[1]:
                logger.debug("daytona: mount %s is ro in spec; uploaded rw "
                             "(no bind mounts on Daytona)", dest)
            c.copy_to(src, dest)
        if ports:
            logger.warning(
                "daytona: host port mapping %s ignored — use preview_link(port) "
                "for external access to a sandbox port", sorted(ports))
    except BaseException:
        # Never leak a sandbox on a partially-failed start: the engine drops the
        # id on failure and would otherwise have no handle left to delete it.
        try:
            c.kill()
        except BaseException:  # noqa: BLE001
            pass
        raise
    return c


class DaytonaContainer(ContainerResource):
    """A Daytona sandbox that is both the resource and its own handle.

    Mirrors :class:`DockerContainer`: subclasses :class:`ContainerResource` and
    passes ``container=self``, so the inherited lifecycle and ``run_cmd`` drive
    the handle methods below.
    """

    def __init__(self, name: str, sandbox: Any, image: Optional[str] = None,
                 workdir: Optional[str] = None,
                 spec: Optional[ContainerResourceSpec] = None):
        self.name = name
        self.sandbox = sandbox
        self.image = image
        self.workdir = workdir
        self.status = "unknown"
        self._killed = False
        super().__init__(container=self, resource_id=name, spec=spec)

    @property
    def sandbox_id(self) -> Optional[str]:
        return getattr(self.sandbox, "id", None)

    # ---- handle (enroot-Container subset, driven by ContainerResource) ---

    def reload(self) -> None:
        if self._killed:
            self.status = "exited"
            return
        try:
            self.sandbox.refresh_data()
        except Exception:  # noqa: BLE001 - a deleted/unreachable sandbox is "exited"
            self.status = "exited"
            return
        state = getattr(self.sandbox, "state", None)
        state = str(getattr(state, "value", state) or "").lower()
        # Daytona reports the VM lifecycle; map it onto docker's vocabulary,
        # which is what ContainerResource.get_status() switches on.
        if state in ("started", "running"):
            self.status = "running"
        elif state in ("stopped", "destroyed", "error", "build_failed", "archived"):
            self.status = "exited"
        else:
            self.status = "creating"

    async def reload_async(self) -> None:
        await asyncio.to_thread(self.reload)

    def kill(self) -> None:
        if self._killed:
            return
        self._killed = True
        self.status = "exited"
        try:
            self.sandbox.delete()
        except Exception as e:  # noqa: BLE001 - teardown must never raise
            logger.warning("daytona: failed to delete sandbox %s: %s",
                           self.sandbox_id, e)

    async def kill_async(self, timeout: int = 10, **_) -> None:
        await asyncio.to_thread(self.kill)

    def preview_link(self, port: int) -> Optional[str]:
        """Public URL for a port inside the sandbox (Daytona's port-mapping
        equivalent), or None when the SDK/plan does not provide one."""
        try:
            link = self.sandbox.get_preview_link(port)
        except Exception:  # noqa: BLE001
            return None
        return getattr(link, "url", None) or str(link)

    # ---- exec -----------------------------------------------------------

    def _exec(self, cmd: Union[str, List[str]], workdir, user,
              environment, timeout) -> DaytonaExecResult:
        command = cmd if isinstance(cmd, str) else shlex.join(cmd)
        if user:
            # No per-exec user flag on Daytona; drop privileges the way the
            # docker `-u` flag would.
            command = f"su {shlex.quote(str(user))} -s /bin/bash -c {shlex.quote(command)}"
        # docker's exec_run folds stderr into stdout (stderr=STDOUT); match it
        # so callers parsing output see the same bytes on both engines.
        command = f"{command} 2>&1"
        try:
            res = self.sandbox.process.exec(
                command,
                cwd=workdir or self.workdir,
                env=dict(environment) if environment else None,
                timeout=int(timeout) if timeout else None,
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if timeout is not None and ("timeout" in msg or "timed out" in msg):
                raise asyncio.TimeoutError(
                    f"daytona exec timed out after {timeout}s") from e
            raise
        out = getattr(res, "result", None)
        if out is None:
            artifacts = getattr(res, "artifacts", None)
            out = getattr(artifacts, "stdout", "") if artifacts else ""
        return DaytonaExecResult(
            exit_code=int(getattr(res, "exit_code", 1) or 0),
            output=(out or "").encode("utf-8", errors="replace"),
        )

    async def exec_run_async(
        self,
        cmd: Union[str, List[str]],
        stdout: bool = True,
        stderr: bool = True,
        demux: bool = False,
        workdir: Optional[str] = None,
        user: Optional[str] = None,
        environment: Optional[dict] = None,
        detach: bool = False,
        timeout: Optional[float] = None,
        **_,
    ) -> DaytonaExecResult:
        return await asyncio.to_thread(
            self._exec, cmd, workdir, user, environment, timeout
        )

    def exec_run(
        self,
        cmd: Union[str, List[str]],
        workdir: Optional[str] = None,
        user: Optional[str] = None,
        environment: Optional[dict] = None,
        timeout: Optional[float] = None,
        **_,
    ) -> DaytonaExecResult:
        return self._exec(cmd, workdir, user, environment, timeout)

    # ---- copy (mirror enroot Container.put_archive / copy_to) ------------

    def put_archive(self, path: str, data: Union[bytes, Any],
                    timeout: float = 300) -> bool:
        """Extract a tar archive into the sandbox at ``path``.

        Daytona has no ``docker cp``; the tar is uploaded to a temp file and
        unpacked in-sandbox, which keeps the same one-round-trip shape and
        avoids per-file upload latency on large trees.
        """
        if hasattr(data, "read"):
            data = data.read()
        remote_tar = f"/tmp/agentfly-{uuid.uuid4().hex[:12]}.tar"
        self.sandbox.fs.upload_file(data, remote_tar, timeout=int(timeout))
        res = self._exec(
            f"mkdir -p {shlex.quote(path)} && "
            f"tar -xf {shlex.quote(remote_tar)} -C {shlex.quote(path)} && "
            f"rm -f {shlex.quote(remote_tar)}",
            None, None, None, timeout,
        )
        if res.exit_code != 0:
            logger.warning("daytona: put_archive into %s failed: %s",
                           path, res.output.decode(errors="replace")[-500:])
        return res.exit_code == 0

    def copy_to(self, local_path: str, container_path: str,
                timeout: float = 300) -> None:
        """Copy a local file/dir into the sandbox.

        Mirrors enroot/docker ``copy_to``: for a directory, its CONTENTS are
        placed at ``container_path``.
        """
        local_path = os.path.abspath(local_path)
        if os.path.isdir(local_path):
            tar_cmd = ["tar", "-C", local_path, "-cf", "-", "."]
        else:
            tar_cmd = ["tar", "-C", os.path.dirname(local_path), "-cf", "-",
                       os.path.basename(local_path)]
        p = subprocess.run(tar_cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, check=True)
        if not self.put_archive(container_path, p.stdout, timeout=timeout):
            raise RuntimeError(f"copy_to failed: {self.name}:{container_path}")

    async def copy_to_async(self, local_path: str, container_path: str,
                            timeout: float = 300) -> None:
        await asyncio.to_thread(self.copy_to, local_path, container_path, timeout)


# --- housekeeping -----------------------------------------------------------

def reap_stale_sandboxes(max_age_min: int = 180) -> int:
    """Delete our own sandboxes older than ``max_age_min``; return the count.

    Scoped by the ``agentfly.managed`` label so it can never touch sandboxes
    created by another tool sharing the account. Best-effort: individual
    failures are logged and skipped.
    """
    from daytona import ListSandboxesQuery

    client = daytona_client()
    deleted = 0
    try:
        # list() yields lazily; materialize before deleting so the iteration
        # isn't walking a collection we are mutating.
        sandboxes = list(client.list(ListSandboxesQuery(labels=dict(_MANAGED_LABEL))))
    except Exception as e:  # noqa: BLE001
        logger.warning("daytona: reap could not list sandboxes: %s", e)
        return 0
    cutoff = time.time() - max_age_min * 60
    for sb in sandboxes:
        created = getattr(sb, "created_at", None)
        try:
            ts = created.timestamp() if hasattr(created, "timestamp") else None
        except Exception:  # noqa: BLE001
            ts = None
        if ts is not None and ts > cutoff:
            continue
        try:
            sb.delete()
            deleted += 1
        except Exception as e:  # noqa: BLE001
            logger.debug("daytona: reap failed for %s: %s",
                         getattr(sb, "id", "?"), e)
    if deleted:
        logger.info("daytona: reaped %d stale sandboxes", deleted)
    return deleted
