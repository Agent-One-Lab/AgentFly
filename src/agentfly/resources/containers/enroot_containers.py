"""The enroot containers AgentFly creates: naming, owner records, and cleanup of orphans.

Every enroot container AgentFly starts (local :class:`~agentfly.resources.runner.LocalRunner`
and the Ray-hosted :class:`~.ray_container_resource.RayEnrootContainerActor`) is named
``agentfly-<resource id>`` (or ``agentfly-<random>`` when no id is given).

**Orphan cleanup.** Containers live in the creating process's resource pool and are reused
across rollouts, so they outlive any single ``agent.run()``. When that process ends without
ending them (a finished script, a crash, ``kill -9``, a Slurm timeout), nothing can reuse them.
To clean those up, AgentFly writes an *owner record* for each container it creates
(:func:`record_container`) and removes it when the container is ended (:func:`forget_container`).
The first time a process creates a container, :func:`sweep_orphaned_containers` runs once in the
background and removes containers whose recorded owner process is gone; ``agentfly train`` also
runs it before training.

The sweep is fail-safe: it removes a container only when it can tell for certain that the owner
is dead (same machine, same PID namespace, and no live process with the owner's PID *and* start
time). Containers it cannot judge, and containers without a record (made by enroot-py directly,
the enroot CLI, or older AgentFly versions), are never touched. Set ``AF_ENROOT_SWEEP=0`` to
turn the sweep off (records are still kept).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ENROOT_CONTAINER_PREFIX = "agentfly"
ENROOT_SWEEP_ENV = "AF_ENROOT_SWEEP"
REMOVE_CONCURRENCY = 4
STOP_TIMEOUT = 10.0


def require_enroot() -> None:
    """Raise a clear error if the ``enroot`` command is missing on this machine.

    Only starting an enroot container needs enroot; importing AgentFly and running agents
    with other tools or engines does not. Called right before a container is started.
    """
    if shutil.which("enroot") is None:
        raise RuntimeError(
            f"Starting an enroot container needs the `enroot` command, which is not installed on "
            f"{socket.gethostname()}. Install enroot (see docs/start/installation.md), or choose another "
            "container engine with AF_CONTAINER_ENGINE=docker, daytona or sqs."
        )


def enroot_container_name(resource_id: Optional[str] = None, prefix: str = ENROOT_CONTAINER_PREFIX) -> str:
    """Return the enroot container name for ``resource_id``: ``<prefix>-<resource_id>``.

    A name that already carries the prefix is returned unchanged, so re-wrapping a
    container name (the resource id of an enroot-backed resource *is* its container
    name) never double-prefixes. Without an id a random ``<prefix>-<hex>`` name is drawn.
    """
    from enroot.utils.utils import random_name

    if not resource_id:
        return random_name(prefix=prefix)
    if resource_id.startswith(prefix + "-"):
        return resource_id
    return f"{prefix}-{resource_id}"


# --------------------------------------------------------------------------- #
# Owner records
# --------------------------------------------------------------------------- #


def _machine_id() -> str:
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        with contextlib.suppress(OSError):
            mid = Path(path).read_text().strip()
            if mid:
                return mid
    return "unknown"


def _pid_namespace() -> str:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return "unknown"


def _process_start_time(pid: int) -> Optional[str]:
    """The kernel's start time of ``pid`` (field 22 of /proc/<pid>/stat), or None if no such process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name (field 2) is in parentheses and may contain spaces; fields after it are plain.
    fields = stat[stat.rfind(")") + 2 :].split()
    return fields[19] if len(fields) > 19 else None


def _identity() -> Dict[str, object]:
    pid = os.getpid()
    return {
        "pid": pid,
        "start_time": _process_start_time(pid),
        "pid_namespace": _pid_namespace(),
        "hostname": socket.gethostname(),
        "machine_id": _machine_id(),
    }


def owners_dir() -> Path:
    """This machine's owner-record folder: ``$AF_CACHE_DIR/enroot/owners/<hostname>-<machine id>``."""
    from ... import AF_CACHE_DIR

    return Path(AF_CACHE_DIR) / "enroot" / "owners" / f"{socket.gethostname()}-{_machine_id()[:12]}"


def _record_path(name: str) -> Path:
    return owners_dir() / f"{name}.json"


def _read_record(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _is_mine(record: dict) -> bool:
    me = _identity()
    return all(record.get(k) == me[k] for k in ("pid", "start_time", "pid_namespace", "machine_id"))


def owner_alive(record: dict) -> Optional[bool]:
    """Whether the record's owner process is alive: True/False, or None if this process can't tell.

    Only records from this machine and this PID namespace can be judged; PIDs from another
    machine or namespace mean nothing here.
    """
    if record.get("machine_id") != _machine_id() or record.get("pid_namespace") != _pid_namespace():
        return None
    pid, start = record.get("pid"), record.get("start_time")
    if not isinstance(pid, int) or not start:
        return None
    return _process_start_time(pid) == start  # a reused PID has a different start time


def record_container(name: str) -> None:
    """Record this process as the owner of container ``name``; call right before creating it.

    Created exclusively, so a live owner's record is never overwritten. If the name is held by a
    dead owner, its leftover container is removed first and the record taken over.
    """
    path = _record_path(name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(_identity(), container=name)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data))
        try:
            os.link(tmp, path)  # atomic and exclusive: fails if a record already exists
        except FileExistsError:
            existing = _read_record(path)
            if existing is not None and not _is_mine(existing):
                if owner_alive(existing) is not False:
                    logger.debug("container %s is recorded for another live owner; not recording", name)
                    return
                _remove_containers([name])  # a dead owner's leftover with the same name
            os.replace(tmp, path)
            tmp = None
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    except OSError as e:  # records are best effort: never block a container start
        logger.debug("could not record owner of container %s: %s", name, e)


def forget_container(name: str) -> None:
    """Drop this process's owner record for ``name``; call once the container is ended."""
    path = _record_path(name)
    record = _read_record(path)
    if record is not None and _is_mine(record):
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #


def _remove_containers(names: List[str]) -> List[str]:
    """Stop the containers' processes, force-remove them, and return the names that are gone."""
    from enroot.utils import (
        container_process_groups,
        list_containers_fancy,
        list_containers_plain,
        run_enroot,
        stop_process_groups,
    )

    listed = list_containers_fancy(strict=False)
    present = [n for n in names if n in listed]
    pids = [pid for n in present for pid in listed[n].get("pids", [])]
    if pids:
        stop_process_groups(container_process_groups(pids), timeout=STOP_TIMEOUT)

    def remove(name: str) -> None:
        try:
            run_enroot(["remove", "--force", name], capture=False, check=False)
        except Exception as e:  # noqa: BLE001 - best effort; a later sweep retries
            logger.debug("enroot remove %s failed: %s", name, e)

    with ThreadPoolExecutor(max_workers=REMOVE_CONCURRENCY) as pool:
        list(pool.map(remove, present))
    remaining = set(list_containers_plain(strict=False))
    return [n for n in names if n not in remaining]


def sweep_orphaned_containers() -> List[str]:
    """Remove this machine's AgentFly containers whose owner process has ended.

    Only containers with an owner record are considered, and only when the owner can be judged
    dead (see :func:`owner_alive`). One sweep runs at a time per machine (a file lock); a process
    that finds the lock taken skips. Never raises. Returns the names of the containers removed.
    """
    if os.environ.get(ENROOT_SWEEP_ENV, "1") == "0":
        return []
    try:
        import fcntl

        folder = owners_dir()
        if not folder.is_dir():
            return []
        with open(folder / ".sweep.lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return []  # another process on this machine is sweeping
            orphans = {}
            for path in folder.glob("*.json"):
                record = _read_record(path)
                if record and record.get("container") and owner_alive(record) is False:
                    orphans[record["container"]] = path
            if not orphans:
                return []
            gone = set(_remove_containers(list(orphans)))
            for name in gone:
                orphans[name].unlink(missing_ok=True)
            removed = sorted(gone)
            if removed:
                logger.info("Removed %d orphaned AgentFly container(s): %s", len(removed), ", ".join(removed))
            return removed
    except Exception as e:  # noqa: BLE001 - cleanup must never break the caller
        logger.debug("orphaned-container sweep skipped: %s", e)
        return []


_sweep_started = False
_sweep_lock = threading.Lock()


def start_background_sweep() -> None:
    """Run :func:`sweep_orphaned_containers` once per process, in a background thread."""
    global _sweep_started
    with _sweep_lock:
        if _sweep_started:
            return
        _sweep_started = True
    threading.Thread(target=sweep_orphaned_containers, name="agentfly-enroot-sweep", daemon=True).start()


def clear_agentfly_containers() -> List[str]:
    """Stop and force-remove **all** of this machine's ``agentfly-*`` enroot containers.

    A manual tool, e.g. for containers left by AgentFly versions that kept no owner records. It
    does not check owners, so only run it when no other AgentFly process uses containers on this
    machine. Never raises; returns the removed names.
    """
    try:
        from enroot import clear_enroot_containers

        removed = clear_enroot_containers(prefix=f"{ENROOT_CONTAINER_PREFIX}-")
    except Exception as e:  # noqa: BLE001
        logger.warning("enroot cleanup skipped: %s", e)
        return []
    with contextlib.suppress(OSError):
        for name in removed:
            _record_path(name).unlink(missing_ok=True)
    return removed
