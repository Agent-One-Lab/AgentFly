"""
udocker_compat.py
A thin wrapper that makes udocker look like docker-py (docker-client) so that
code written for `docker.from_env()` keeps working on locked-down machines
where only udocker is available.

Limitations
-----------
* Only the most commonly-used methods are implemented.
* Many advanced flags are accepted but silently ignored.
* exec/attach are synchronous; there is no streaming API.
"""
from __future__ import annotations
import asyncio
import hashlib
import re
import json, subprocess, shlex, os, pathlib, sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid
# ─── helpers ────────────────────────────────────────────────────────────────
import socket, random

def _allocate_free_port() -> str:
    """Ask the kernel for a free TCP port and return it as a string."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])
    

def _parse_ports_kw(ports_kw: dict[str, Any] | None) -> tuple[dict[str, str], list[str]]:
    """
    Translate docker-py's  ports={"8000/tcp": None, "9000/tcp": 12345}
    → (internal port-map dict,
       list of '-p host:guest' flags for udocker)
    """
    if not ports_kw:
        return {}, []

    port_map: dict[str, str] = {}
    flags:    list[str]      = []

    for cport, host in ports_kw.items():
        # container part comes before the '/', e.g. '8000/tcp'
        guest = cport.split("/")[0]

        if host is None:
            host = _allocate_free_port()
        elif isinstance(host, (list, tuple)):
            host = str(host[0])
        else:
            host = str(host)

        port_map[f"{guest}/tcp"] = host
        flags.extend([f"--publish={host}:{guest}"])

    return port_map, flags


async def a_run_udocker(
    args: List[str],
    capture: bool = True,
    check: bool = True,
) -> str | None:
    """
    Asynchronous equivalent of _run_udocker().
    Uses asyncio.create_subprocess_exec so the event-loop is never blocked.
    """
    # assemble exec
    proc = await asyncio.create_subprocess_exec(
        "udocker", *map(str, args),
        stdout=asyncio.subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE if capture else subprocess.DEVNULL,
    )

    # read output concurrently with process exit
    out_bytes, err_bytes = await proc.communicate()
    if check and proc.returncode:
        raise RuntimeError(
            f"$ udocker {' '.join(args)}\n{err_bytes.decode()}"
        )
    return None if not capture else out_bytes.decode().strip()


def _run_udocker(args, **kw):
    """
    If called from sync code → run normally (blocking).
    If called while an event-loop is running → delegate to a_run_udocker().
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # we're in async context – run non-blocking
        return loop.run_until_complete(a_run_udocker(args, **kw))
    else:
        # original blocking implementation
        res = subprocess.run(
            ["udocker", *map(str, args)],
            stdout=subprocess.PIPE if kw.get("capture", True) else None,
            stderr=subprocess.PIPE if kw.get("capture", True) else None,
            text=True,
        )
        if kw.get("check", True) and res.returncode and res.stderr:
            raise RuntimeError(f"$ udocker {' '.join(args)}\n{res.stderr}")
        return res.stdout.strip() if kw.get("capture", True) else None



# ---------- ps parsing -------------------------------------------------------
_PS_SPLIT = re.compile(r"\s{2,}")          # collapse ≥2 blanks

# ------------------------------------------------------------------ helpers
def _parse_ps(text: str) -> list[str]:
    """
    Return the list of container-IDs shown by `udocker ps`.

    The first whitespace-separated token of every non-header line
    *is* the UUID-like container id.

        00beea81-db15-3eef-890f-b869010c5edc . W  ubuntu:latest
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    """
    lines = text.splitlines()

    # skip header: starts with 'CONTAINER ID'
    if lines and lines[0].lstrip().startswith("CONTAINER ID"):
        lines = lines[1:]

    cids: list[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        cid = ln.split(None, 1)[0]    # first token (whitespace 1+)
        cids.append(cid)
    return cids

# ▲ CHANGED ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
_SHA256_RGX = re.compile(r"^/sha256:([0-9a-f]{64})")

def _hash_tag(tag: str) -> str:
    """Fallback pseudo-ID (12-char hex) if no sha256 found."""
    return hashlib.sha256(tag.encode()).hexdigest()[:12]

def _parse_images(text: str) -> List[Image]:
    """
    Parse the *non-table* format printed by `udocker images` or `images -l`.

    Example chunk:

        REPOSITORY
        python:3.11-alpine    .
         /home/user/.udocker/repos/python/3.11-alpine
            /sha256:a78edf41f9ae...   (15 MB)
    """
    images: List[Image] = []
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # skip the header
        if line == "REPOSITORY":
            i += 1
            continue

        # repository-tag line → first token before whitespace
        if ":" in line:
            tag = line.split()[0]        # e.g. python:3.11-alpine
            sha256 = None

            # look ahead for the first "/sha256:<digest>" child line
            j = i + 1
            while j < len(lines) and not lines[j].startswith("REPOSITORY"):
                m = _SHA256_RGX.match(lines[j].lstrip())
                if m:
                    sha256 = m.group(1)[:12]  # first 12 chars = docker-style short ID
                    break
                # stop if we hit the next repo entry
                if ":" in lines[j] and lines[j].endswith("."):
                    break
                j += 1

            img_id = sha256 or _hash_tag(tag)
            images.append(Image(id=img_id, tags=[tag]))
            i = j
        else:
            i += 1
    return images

def _name_opt(name: str | None) -> list[str]:
    """Return ['--name=<name>'] if a name was supplied, else []"""
    return [f"--name={name}"] if name else []


@dataclass
class Image:
    id: str
    tags: List[str]

class Images:
    """docker-py: client.images"""
    def __init__(self, client: "UdockerClient"):
        self._cli = client

    # docker-py: images.pull(repository, tag=None, **kwargs)
    def pull(self, repository: str, tag: str | None = None, **_) -> Image:
        ref = f"{repository}:{tag}" if tag else repository
        _run_udocker(["pull", ref], capture=False)
        return self.get(ref)

    # docker-py: images.list()
    def list(self, **_) -> List[Image]:
        text = _run_udocker(["images", "-l"])  # ▲ CHANGED
        return _parse_images(text)

    # docker-py: images.get(name_or_id)
    def get(self, ref: str) -> Image:
        # quick path: look at list() output first
        for im in self.list():
            if ref in (im.id, *im.tags):
                return im
        # fallback: inspect (may raise)
        raw = _run_udocker(["inspect", ref])
        return Image(id=ref, tags=[ref])  # minimal fallback



class Container:
    """
    docker-py-like container object backed by udocker CLI.
    """
    def __init__(self, client: "UdockerClient",
                 cid: str,
                 name: str | None,
                 port_map: dict[str, str] | None = None):
        self._cli   = client
        self.id     = cid
        self.name   = name or cid
        self._port_map = port_map or {}            # ← NEW
        self._status: str = "exited"
        self._attrs: dict = {}

    # ---------------------------------------------------------------- status
    @property
    def status(self):          # running | exited
        return self._status

    def reload(self):
        txt     = _run_udocker(["ps"])
        running = self.id in _parse_ps(txt)
        self._status = "running" if running else "exited"

        # assemble docker-style attrs exactly once
        if not self._attrs:
            self._attrs = {
                "NetworkSettings": {
                    "Ports": {
                        cport: [{
                            "HostIp":  "127.0.0.1",
                            "HostPort": hport,
                        }]
                        for cport, hport in self._port_map.items()
                    }
                }
            }

    @property
    def attrs(self):
        if not self._attrs:
            self.reload()
        return self._attrs

    def kill(self, **_):
        _run_udocker(["rm", "-f", self.id], capture=False)
        self._status = "exited"

    def exec_run(self, cmd: str | List[str], **_) -> "ExecResult":
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        out = _run_udocker(["exec", self.id, *cmd])
        return ExecResult(exit_code=0, output=out)

    # docker-py: container.remove(force=True)
    def remove(self, force: bool = False, **_):
        _run_udocker(["rm", "-f" if force else "", self.id], capture=False)

    # docker-py: container.logs()
    def logs(self, **_) -> str:
        path = (
            pathlib.Path.home()
            / ".udocker"
            / "containers"
            / self.id
            / "log"
            / "json.log"
        )
        log_text = path.read_text() if path.exists() else ""
        log_text = log_text.encode("utf-8")
        return log_text

    # docker-py: container.wait()
    def wait(self, **_) -> Dict[str, Any]:
        proc_info = self.inspect()
        return {"StatusCode": proc_info.get("ExitCode", 0)}

    def inspect(self) -> Dict[str, Any]:
        raw = _run_udocker(["inspect", "--json", self.id])
        return json.loads(raw)

@dataclass
class ExecResult:
    exit_code: int
    output: str


class Containers:
    """ Mimics docker.DockerClient.containers """
    def __init__(self, client: "UdockerClient"):
        self._cli = client

    def create(self, image, command=None, name=None, **_):
        cmd = shlex.split(command) if isinstance(command, str) else (command or [])
        cid = _run_udocker(
            ["create", *_name_opt(name), image, *cmd]
        )
        return Container(self._cli, cid, name)

    def run(self, image, command=None, name=None,
            detach=False, remove=False, **kwargs):

        cmd  = shlex.split(command) if isinstance(command, str) else (command or [])
        ports_kw = kwargs.pop("ports", {})            # ← docker-py kw
        port_map, port_flags = _parse_ports_kw(ports_kw)

        # ---- DETACH ------------------------------------------------------
        if detach:
            # 1) create container to obtain CID
            create_args = ["create", *_name_opt(name), image, *cmd]
            cid = _run_udocker(create_args)

            # 2) start it in background
            subprocess.Popen(
                ["udocker", "run", *port_flags, cid],               # run by CID
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return Container(self._cli, cid, name, port_map)

        # ---- foreground run ---------------------------------------------
        run_args = ["run", *_name_opt(name), *port_flags,
                    *(["--rm"] if remove else []), image, *cmd]
        return _run_udocker(run_args)


    # docker-py: .get(container_id_or_name)
    def get(self, ident: str) -> Container:
        info = _run_udocker(["inspect", "--json", ident])
        meta = json.loads(info)
        return Container(self._cli, ident, meta.get("Name"))

    def list(self, all: bool = False, **_) -> List["Container"]:
        txt   = _run_udocker(["ps"])          # ← flag-free call
        cids  = _parse_ps(txt)
        return [Container(self._cli, cid, None) for cid in cids]

    # helper -----------------------------------------------------------
    def _latest_container_id(self) -> str:
        lines = _run_udocker(["ps", "-r"]).splitlines()
        if not lines:
            raise RuntimeError("No running udocker containers found")
        # first column is CID
        return lines[0].split()[0]
    

class UdockerClient:
    """
    A compatibility facade that follows the public API of
    `docker.DockerClient` (`docker.from_env()`).

    Only a subset needed for most ML / RL workflows is implemented.
    """
    def __init__(self):
        self.images = Images(self)
        self.containers = Containers(self)

    # docker-py: .ping()
    def ping(self) -> bool:
        try:
            _run_udocker(["version"])
            return True
        except Exception:
            return False

    # Low-level alias for docker-py compatibility (some code uses .api)
    @property
    def api(self) -> "UdockerClient":
        return self

    # docker-py: .close() – no-op for udocker
    def close(self):
        pass

# ------------------------------------------------------------------ public API
def from_env() -> UdockerClient:  # mirrors docker.from_env()
    return UdockerClient()
    
    
async def kill_all_containers():
    cids = _parse_ps(await a_run_udocker(["ps"]))
    for cid in cids:
        asyncio.create_task(a_run_udocker(["rm", cid], capture=False))
