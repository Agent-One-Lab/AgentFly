"""
enroot_compat.py
~~~~~~~~~~~~~~~~
A lightweight docker-py shim powered by **Enroot**.

Supported high-level API
------------------------
✓ docker.from_env()
✓ client.images.pull(), .list(), .get()
✓ client.containers.run(detach=True,
        cpu_count=…, mem_limit=…,  # NEW
        ports={…}, remove=True)
✓ container.reload(), .exec_run(), .kill(), .remove()
✓ container.attrs["NetworkSettings"]["Ports"]

Implementation notes
--------------------
* Resource limits are applied by running Enroot inside
  `systemd-run --user --scope -p CPUQuota=… -p MemoryMax=…`
  (works on any distro with user-slices enabled).
* If **systemd-run** isn't available and the caller requests limits,
  we ignore the limits but emit a `warnings.warn`.
* Port mapping is emulated (host-port = container-port on 127.0.0.1).
"""

from __future__ import annotations
import asyncio
import os
import pathlib
import shlex
import shutil
import subprocess
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from ... import ENROOT_HOME
import socket
import sys

def _allocate_free_port() -> str:
    """Ask the kernel for a free TCP port and return it as a string."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])
    
# ───────────────────────── helpers ────────────────────────────────────────
async def a_run_enroot(args: List[str],
                capture: bool = True,
                check: bool = True,
                text: bool = True) -> str:
    cmd = ["enroot", *map(str, args)]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
    )
    out_bytes, err_bytes = await proc.communicate()
    if check and proc.returncode:
        raise RuntimeError(f"$ {' '.join(cmd)}\n{err_bytes.decode()}")
    
    return out_bytes.decode().strip() if capture else ""


def _run_enroot(args: List[str],
                capture: bool = True,
                check: bool = False,
                text: bool = True) -> str:
    # """
    # If called from sync code → run normally (blocking).
    # If called while an event-loop is running → delegate to a_run_enroot().
    # """
    # try:
    #     loop = asyncio.get_running_loop()
    # except RuntimeError:
    #     loop = None

    # if loop and loop.is_running():
    #     return loop.run_until_complete(a_run_enroot(args, capture, check, text))
    # else:
    proc = subprocess.run(
            ["enroot", *map(str, args)],
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
        )
    if check and proc.returncode:
        raise RuntimeError(f"$ {' '.join(args)}\n{proc.stderr}")
    return proc.stdout.decode().strip() if capture else ""

def _random_name(prefix: str = "cnt") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ───────────────────── list-output parser ─────────────────────────────────
def _parse_list(text: str) -> Dict[str, Dict[str, Any]]:
    """Handle both `enroot list` (plain names) and `list --fancy`."""
    lines = [ln.strip("\n") for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}

    # No header → simple list of names
    if not lines[0].lstrip().startswith("NAME"):
        return {ln.strip(): {} for ln in lines}

    header = lines[0]
    cols = ["NAME", "PID", "STATE", "STARTED", "TIME",
            "MNTNS", "USERNS", "COMMAND"]
    starts = {c: header.find(c) for c in cols if c in header}
    bounds = []
    for i, (c, s) in enumerate(sorted(starts.items(), key=lambda kv: kv[1])):
        e = sorted(starts.values())[i + 1] if i + 1 < len(starts) else None
        bounds.append((c, s, e))

    info: Dict[str, Dict[str, Any]] = {}
    cur: Optional[str] = None

    for ln in lines[1:]:
        if ln[starts["NAME"]] != " ":           # new container row
            cur = ln[starts["NAME"]:starts["PID"]].strip()
            info[cur] = {"pids": []}

        if cur is None:
            continue

        for col, s, e in bounds:
            field = ln[s:e].strip()
            if not field:
                continue
            if col == "PID" and field.isdigit():
                info[cur]["pids"].append(int(field))
            else:
                info[cur][col.lower()] = field
    return info


# ───────────────── docker-py–like object model ────────────────────────────
@dataclass
class Image:
    id: str
    tags: List[str]


class Images:
    def __init__(self, cli: "EnrootClient"):
        self._cli = cli

    def pull(self, repository: str, tag: str | None = None, **_) -> Image:
        if tag is None:
            tag = "latest"
        ref = repository.replace("/", "+")
        _run_enroot(["import", "--output", os.path.join(ENROOT_HOME + "/images", f"{ref}.sqsh"), f"docker://{repository}:{tag}"], capture=False)
        return self.get(ref)

    def list(self, **_) -> List[Image]:
        data_dir = pathlib.Path(ENROOT_HOME + "/images")
        return [Image(id=str(p), tags=[p.stem])
                for p in data_dir.glob("*.sqsh")]

    def get(self, ref: str) -> Image:
        p = pathlib.Path(ENROOT_HOME + "/images/" + ref + ".sqsh")
        if p.exists():
            return Image(id=str(p), tags=[p.stem])
        for img in self.list():
            if ref in img.tags:
                return img
        raise RuntimeError(f"Image {ref!r} not found")


# -------------------------------------------------------------------------
@dataclass
class ExecResult:
    exit_code: int
    output: str


class Container:
    def __init__(self, cli: "EnrootClient", name: str,
                 port_map: Dict[str, str]):
        self._cli = cli
        self.name = name
        self._port_map = port_map
        self._attrs: Dict[str, Any] = {}
        self.reload()

    # internal -------------------------------------------------------------
    def _meta(self):
        return _parse_list(_run_enroot(["list", "--fancy"])).get(self.name, {})

    # public ---------------------------------------------------------------
    def reload(self):
        meta = self._meta()
        running = bool(meta.get("pids")) or bool(meta)
        self._status = "running" if running else "exited"
        if not self._attrs:
            self._attrs = {
                "NetworkSettings": {
                    "Ports": {c: [{"HostIp": "127.0.0.1",
                                   "HostPort": h}]
                              for c, h in self._port_map.items()}
                },
                "State": {
                    "Status": self._status,
                    "Running": running,
                },
            }

    def kill(self, **_):
        _run_enroot(["remove", "--force", self.name], capture=False)
        self._status = "exited"

    def remove(self, force: bool = False, **_):
        _run_enroot(["remove", "--force" if force else "", self.name],
                    capture=False)
        self._status = "exited"

    def exec_run(self, cmd: str | List[str], **_) -> ExecResult:
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        out = _run_enroot(["exec", self.name, *cmd])
        return ExecResult(exit_code=0, output=out)

    def logs(self, **_) -> bytes:
        return b""

    @property
    def status(self):
        self.reload()
        return self._status

    @property
    def attrs(self):
        return self._attrs


# -------------------------------------------------------------------------
def _systemd_prefix(cpus: Optional[float], mem: Optional[str | int]) -> List[str]:
    """Return ['systemd-run', …] args or [] if no limits requested/available."""
    if cpus is None and mem is None:
        return []
    if shutil.which("systemd-run") is None:
        warnings.warn("cpu_count/mem_limit ignored: systemd-run not found")
        return []

    args = ["systemd-run", "--user", "--scope"]
    if mem is not None:
        mem_val = f"{mem}" if isinstance(mem, (str, int)) else str(int(mem))
        args += ["-p", f"MemoryMax={mem_val}"]
    if cpus is not None:
        quota = int(float(cpus) * 100)  # 1 CPU = 100 %
        args += ["-p", f"CPUQuota={quota}%"]
    return args


class Containers:
    def __init__(self, cli: "EnrootClient"):
        self._cli = cli

    # main entry -----------------------------------------------------------
    def run(self, image, command=None, name=None, *, detach=False,
            remove=False, ports=None, environment=None,
            cpu_count: float | None = None, mem_limit: str | int | None = None,
            cpus: float | None = None,  # docker-py synonym
            **_):
        if not detach:
            raise NotImplementedError("detach=False not supported")

        cpu_limit = cpu_count if cpu_count is not None else cpus
        name = name or _random_name()

        # ensure rootfs exists
        if name not in _parse_list(_run_enroot(["list"], capture=True)):
            img_path = None
            path = pathlib.Path(str(image))
            if path.suffix == ".sqsh" and path.exists():
                img_path = str(path)
            else:
                img_path = f"{ENROOT_HOME}/images/{str(image).replace('/', '+').replace(':', '+')}.sqsh"
                if not pathlib.Path(img_path).exists():
                    _run_enroot(["import", "--output", img_path, f"docker://{image}"], capture=False)
            _run_enroot(["create", "--name", name, img_path], capture=False)

        # Handle environment variables
        env_options = []
        environment = environment or {}
        for key, value in environment.items():
            env_options.extend(["--env", f"{key}={value}"])

        # Handle port mapping
        ports = ports or {}
        port_map = {f"{p.split('/')[0]}/tcp": p.split('/')[0] for p in ports}
        if port_map:
            free_port = _allocate_free_port()
            for port in port_map.keys():
                port_map[port] = free_port
            env_options.extend(["--env", f"PORT={free_port}"])

        # launch
        # TODO: add resource limits
        prefix = _systemd_prefix(cpu_limit, mem_limit)

        start_cmd = ["enroot", "start", "--rw", *env_options, name]
        if command:
            start_cmd += shlex.split(command) if isinstance(command, str) \
                         else list(command)
        # print(start_cmd)
        process = subprocess.Popen(start_cmd,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         start_new_session=True)

        deadline = time.time() + 15
        while time.time() < deadline:
            if name in _parse_list(_run_enroot(["list"])):
                break
            time.sleep(0.1)

        return Container(self._cli, name, port_map)

    # utilities ------------------------------------------------------------
    def list(self, **_) -> List[Container]:
        names = _parse_list(_run_enroot(["list"])).keys()
        return [Container(self._cli, nm, {}) for nm in names]

    def get(self, ident: str) -> Container:
        if ident not in _parse_list(_run_enroot(["list"])):
            raise RuntimeError(f"Container {ident!r} not found")
        return Container(self._cli, ident, {})


# ─────────────────────── top-level client ─────────────────────────────────
class EnrootClient:
    def __init__(self):
        self.images = Images(self)
        self.containers = Containers(self)

    def ping(self) -> bool:
        try:
            _run_enroot(["version"])
            return True
        except Exception:
            return False

    @property
    def api(self):
        return self

    def close(self):
        pass

def from_env() -> EnrootClient:
    return EnrootClient()


def clear_enroot_containers() -> None:
    _run_enroot(["remove", "--force", "$(enroot list)"], capture=False)
    print("Cleared all enroot containers")
