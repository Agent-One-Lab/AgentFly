"""Docker-backed container resource.

:class:`DockerContainer` is a :class:`ContainerResource` (and thus a
:class:`BaseResource`): the docker backend returns it directly to the resource
engine, no separate wrapper. It is its own container handle — it implements the
enroot-``Container`` subset that :class:`ContainerResource` drives, and passes
``container=self`` to the base, so the inherited ``run_cmd`` / ``start`` /
``get_status`` / ``end`` / ``close`` lifecycle calls back into these methods:

    - ``name``                  attribute (and ``resource_id`` == ``name``)
    - ``status``                attribute (refreshed by ``reload``)
    - ``reload()`` / ``reload_async()``
    - ``exec_run_async(cmd, workdir=, user=, environment=, timeout=)`` ->
      object with ``.exit_code`` and ``.output`` (bytes, stdout+stderr)
    - ``kill_async()``

Backed by the ``docker`` CLI, so it works against a local daemon or a
remote one via ``DOCKER_HOST=ssh://user@host`` transparently.

Bind mounts are resolved daemon-side, so when the daemon is remote the
mount *source* must live on the remote host. ``start_docker_container``
detects an ``ssh://`` ``DOCKER_HOST`` and rsyncs each local mount source to
a staging dir on the remote before mounting it. With a local daemon the
sources are mounted directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ..types import ContainerResourceSpec
from .container_resource import ContainerResource


@dataclass
class DockerExecResult:
    """docker-py / enroot ExecResult-compatible: (exit_code, output)."""
    exit_code: int
    output: bytes

    def __iter__(self):
        return iter((self.exit_code, self.output))

    def __getitem__(self, i):
        return (self.exit_code, self.output)[i]


def _docker_bin() -> str:
    return os.environ.get("DOCKER_BIN", "docker")


def docker_hosts_pool() -> List[str]:
    """The DOCKER_HOST pool from the ``DOCKER_HOSTS`` env var.

    ``DOCKER_HOSTS`` is a comma/whitespace-separated list of docker host URLs
    (e.g. ``ssh://ubuntu@a,ssh://ubuntu@b``), mirroring the single-host
    ``DOCKER_HOST``. Empty/unset -> ``[]`` (single-host mode).
    """
    raw = os.environ.get("DOCKER_HOSTS", "").strip()
    if not raw:
        return []
    return [h.strip() for h in re.split(r"[,\s]+", raw) if h.strip()]


def select_docker_host(key: str) -> Optional[str]:
    """Map ``key`` to one host in the ``DOCKER_HOSTS`` pool, deterministically.

    The same key always resolves to the same host, with no shared state, so
    independent code paths (the image build, the container run, the grader)
    agree on a host just by hashing the same key — no coordination needed.
    Key on the IMAGE TAG: images are per-daemon, so a tag's build and every
    container started from it MUST land on the same host. Returns ``None`` when
    no pool is configured, so callers fall back to the single ``DOCKER_HOST``.
    """
    hosts = docker_hosts_pool()
    if not hosts:
        return None
    # md5, not the builtin hash(): hash() of str is salted per process, so two
    # processes (or runs) would disagree on the host for the same key.
    digest = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return hosts[digest % len(hosts)]


def _ssh_target_from_docker_host(docker_host: Optional[str] = None) -> Optional[str]:
    """If the docker host is ssh://user@host[:port], return user@host[:port].

    Uses ``docker_host`` when given (a pool selection), else ``DOCKER_HOST``.
    """
    val = (docker_host or os.environ.get("DOCKER_HOST", "")).strip()
    m = re.match(r"^ssh://([^/]+)(/.*)?$", val)
    return m.group(1) if m else None


def _run(cmd: List[str], timeout: Optional[float] = None,
         input_bytes: Optional[bytes] = None,
         docker_host: Optional[str] = None) -> subprocess.CompletedProcess:
    # When a pool host is selected, target it by overriding DOCKER_HOST for this
    # one call only — the process-global env stays the single-host fallback.
    env = {**os.environ, "DOCKER_HOST": docker_host} if docker_host else None
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        input=input_bytes,
        timeout=timeout,
        env=env,
    )


def _translate_mount_opts(opts: str) -> str:
    """Map enroot bind opts to docker -v opts.

    enroot uses things like ``ro,rbind`` / ``rw,rbind``; docker only
    understands ``ro`` / ``rw`` (rbind is implicit). Keep ro/rw; drop the
    rest. Default rw when nothing recognizable is present.
    """
    parts = {p.strip() for p in opts.split(",") if p.strip()}
    if "ro" in parts:
        return "ro"
    return "rw"


def _stage_mount_to_remote(target: str, src: str, container_name: str) -> str:
    """rsync a local mount source to a staging dir on the remote docker host.

    Returns the remote path to mount. Directories and files are both
    handled. Raises RuntimeError on failure.
    """
    base = src.rstrip("/")
    leaf = os.path.basename(base) or "mount"
    remote_dir = f"~/.agentfly-mounts/{container_name}"
    # Ensure the staging dir exists.
    rc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", target, f"mkdir -p {remote_dir}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if rc.returncode != 0:
        raise RuntimeError(f"mkdir on remote failed: {rc.stderr.decode(errors='replace')}")
    is_dir = os.path.isdir(src)
    # Trailing slash semantics: copy the dir/file as <leaf> under remote_dir.
    rsync_src = base + ("/" if is_dir else "")
    remote_leaf = f"{remote_dir}/{leaf}" + ("/" if is_dir else "")
    if is_dir:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", target, f"mkdir -p {remote_dir}/{leaf}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    proc = subprocess.run(
        ["rsync", "-az", "-e", "ssh -o BatchMode=yes",
         rsync_src, f"{target}:{remote_leaf}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"rsync of mount {src!r} to remote failed: "
            f"{proc.stderr.decode(errors='replace')}"
        )
    # The remote path (expand ~ to an absolute path for docker -v).
    # docker -v needs an absolute path; resolve ~ on the remote.
    home = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", target, "printf %s \"$HOME\""],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.decode().strip()
    rel = f".agentfly-mounts/{container_name}/{leaf}"
    return f"{home}/{rel}" if home else f"{remote_dir}/{leaf}"


def _build_mount_args(mount: Dict[str, str], container_name: str,
                      docker_host: Optional[str] = None) -> List[str]:
    """Translate spec.mount into docker -v args, staging to remote if needed."""
    if not mount:
        return []
    target = _ssh_target_from_docker_host(docker_host)
    args: List[str] = []
    for src, dest_spec in mount.items():
        # dest_spec is like "/container/path:ro,rbind" or "/container/path"
        if ":" in dest_spec:
            dest, opts = dest_spec.split(":", 1)
            docker_opt = _translate_mount_opts(opts)
        else:
            dest, docker_opt = dest_spec, "rw"
        host_src = src
        if target is not None and os.path.exists(src):
            host_src = _stage_mount_to_remote(target, src, container_name)
        args += ["-v", f"{host_src}:{dest}:{docker_opt}"]
    return args


def start_docker_container(
    image: str,
    name: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    mount: Optional[Dict[str, str]] = None,
    ports: Optional[Dict[str, Any]] = None,
    workdir: Optional[str] = None,
    timeout: Optional[float] = None,
    docker_host: Optional[str] = None,
    spec: Optional["ContainerResourceSpec"] = None,
) -> "DockerContainer":
    """``docker run -d ... sleep infinity`` and return a handle.

    The container is kept alive with ``sleep infinity`` so subsequent
    ``exec`` calls have a persistent place to run, matching the enroot
    long-lived-container model.

    With a ``DOCKER_HOSTS`` pool configured, the container (and every later
    ``exec``/``cp`` against it) is pinned to one host, chosen from the image
    tag so it lands on the same daemon the image was built on. ``docker_host``
    overrides that selection; ``None`` falls back to the single ``DOCKER_HOST``.
    """
    # Placement precedence: explicit arg -> spec.docker_host (caller-pinned) ->
    # image-tag hash (the default). The middle term lets a multi-step task pin
    # all its containers to one host even when their image tags differ.
    docker_host = (docker_host
                   or (getattr(spec, "docker_host", None) if spec else None)
                   or select_docker_host(image))
    cname = name or f"agentfly-{uuid.uuid4().hex[:12]}"
    cmd = [_docker_bin(), "run", "-d", "--name", cname]
    if workdir:
        cmd += ["-w", workdir]
    for k, v in (environment or {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd += _build_mount_args(mount or {}, cname, docker_host)
    for hostp, ctrp in (ports or {}).items():
        cmd += ["-p", f"{hostp}:{ctrp}"]
    cmd += [image, "sleep", "infinity"]

    # CLEANUP ON FAILURE: a `docker run` that fails AFTER reserving the name (a
    # non-zero exit leaving a Created-state container, or a timeout while the
    # daemon still created it under heavy load) leaves an orphan holding `cname`.
    # The engine drops the id from its map on failure but can't remove the
    # container (it has no handle), so the NEXT acquire for the same rollout id
    # re-runs `docker run --name cname` and hits "name already in use" — which
    # cascades into every tool call failing and the task being wrongly skipped.
    # Removing the orphan here, before re-raising, lets the engine's retry succeed.
    try:
        proc = _run(cmd, timeout=timeout, docker_host=docker_host)
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker run failed for image={image!r} name={cname!r}: "
                f"{(proc.stdout or b'').decode(errors='replace')[-1000:]}"
            )
        c = DockerContainer(cname, image=image, docker_host=docker_host, spec=spec)
        c.reload()
    except BaseException:
        try:
            _run([_docker_bin(), "rm", "-f", cname], timeout=timeout,
                 docker_host=docker_host)
        except BaseException:
            pass
        raise
    return c


class DockerContainer(ContainerResource):
    """A docker container that is both the resource and its own handle.

    Subclasses :class:`ContainerResource` (a :class:`BaseResource`) and passes
    ``container=self``, so the inherited lifecycle (``start``/``get_status``/
    ``end``/``close``) and ``run_cmd`` drive the docker handle methods below.
    """

    def __init__(self, name: str, image: Optional[str] = None,
                 docker_host: Optional[str] = None,
                 spec: Optional["ContainerResourceSpec"] = None):
        self.name = name
        self.image = image
        # Pin every docker call for this container to the host it was created
        # on, so exec/cp/inspect/rm hit the daemon that actually holds it.
        self.docker_host = docker_host
        self.status = "unknown"
        # We are our own handle: ContainerResource drives reload/exec/kill on
        # ``container``, and resource_id is the container name.
        super().__init__(container=self, resource_id=name, spec=spec)

    # ---- handle (enroot-Container subset, driven by ContainerResource) ---

    def reload(self) -> None:
        proc = _run([_docker_bin(), "inspect", "-f", "{{.State.Status}}", self.name],
                    docker_host=self.docker_host)
        if proc.returncode != 0:
            self.status = "exited"
        else:
            self.status = (proc.stdout or b"").decode(errors="replace").strip() or "exited"

    async def reload_async(self) -> None:
        await asyncio.to_thread(self.reload)

    def kill(self) -> None:
        _run([_docker_bin(), "rm", "-f", self.name], docker_host=self.docker_host)

    async def kill_async(self, timeout: int = 10, **_) -> None:
        await asyncio.to_thread(self.kill)
        # Best-effort cleanup of any remote mount staging.
        target = _ssh_target_from_docker_host(self.docker_host)
        if target is not None:
            await asyncio.to_thread(
                lambda: subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", target,
                     f"rm -rf ~/.agentfly-mounts/{self.name}"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            )

    # ---- exec -----------------------------------------------------------

    def _exec(self, cmd: Union[str, List[str]], workdir, user,
              environment, timeout) -> DockerExecResult:
        argv = [_docker_bin(), "exec"]
        if workdir:
            argv += ["-w", workdir]
        if user:
            argv += ["-u", str(user)]
        for k, v in (environment or {}).items():
            argv += ["-e", f"{k}={v}"]
        argv.append(self.name)
        if isinstance(cmd, str):
            argv += ["bash", "-c", cmd]
        else:
            argv += list(cmd)
        proc = _run(argv, timeout=timeout, docker_host=self.docker_host)
        return DockerExecResult(exit_code=proc.returncode, output=proc.stdout or b"")

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
    ) -> DockerExecResult:
        try:
            return await asyncio.to_thread(
                self._exec, cmd, workdir, user, environment, timeout
            )
        except subprocess.TimeoutExpired as e:
            raise asyncio.TimeoutError(
                f"docker exec timed out after {timeout}s"
            ) from e

    def exec_run(
        self,
        cmd: Union[str, List[str]],
        workdir: Optional[str] = None,
        user: Optional[str] = None,
        environment: Optional[dict] = None,
        timeout: Optional[float] = None,
        **_,
    ) -> DockerExecResult:
        try:
            return self._exec(cmd, workdir, user, environment, timeout)
        except subprocess.TimeoutExpired as e:
            raise asyncio.TimeoutError(
                f"docker exec timed out after {timeout}s"
            ) from e

    # ---- copy (mirror enroot Container.put_archive / copy_to) ------------

    def put_archive(self, path: str, data: Union[bytes, Any], timeout: float = 300) -> bool:
        """Extract a tar archive into the container at ``path`` (docker cp -).

        Mirrors enroot ``Container.put_archive`` so both backends share one
        interface. ``path`` is created first; ``data`` is a tar byte stream.
        """
        if hasattr(data, "read"):
            data = data.read()
        self._exec(f"mkdir -p {shlex.quote(path)}", None, None, None, 60)
        proc = _run([_docker_bin(), "cp", "-", f"{self.name}:{path}"],
                    input_bytes=data, timeout=timeout, docker_host=self.docker_host)
        return proc.returncode == 0

    def copy_to(self, local_path: str, container_path: str, timeout: float = 300) -> None:
        """Copy a local file/dir into the running container.

        Mirrors enroot ``Container.copy_to``: for a directory, its CONTENTS are
        placed at ``container_path``. Builds a tar on the host and forwards it
        to :meth:`put_archive`.
        """
        local_path = os.path.abspath(local_path)
        if os.path.isdir(local_path):
            tar_cmd = ["tar", "-C", local_path, "-cf", "-", "."]
        else:
            tar_cmd = ["tar", "-C", os.path.dirname(local_path), "-cf", "-",
                       os.path.basename(local_path)]
        p = subprocess.run(tar_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if not self.put_archive(container_path, p.stdout, timeout=timeout):
            raise RuntimeError(f"copy_to failed: {self.name}:{container_path}")

    async def copy_to_async(self, local_path: str, container_path: str,
                            timeout: float = 300) -> None:
        await asyncio.to_thread(self.copy_to, local_path, container_path, timeout)
