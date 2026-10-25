"""AgentFly-owned enroot containers: ``agentfly-`` naming + the startup sweep hook.

The sweep itself is ``enroot.clear_enroot_containers`` (tested in enroot-py); here we
check that AgentFly names its containers with the prefix and that the pre-training
sweep calls the library with that prefix.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("enroot", reason="enroot client required")

from agentfly.resources.containers.enroot_containers import (  # noqa: E402
    ENROOT_CONTAINER_PREFIX,
    enroot_container_name,
)


# ---- naming -------------------------------------------------------------------

def test_prefix_is_agentfly():
    assert ENROOT_CONTAINER_PREFIX == "agentfly"


def test_name_prefixes_resource_id():
    assert enroot_container_name("abc-123") == "agentfly-abc-123"


def test_name_is_idempotent_for_already_prefixed_ids():
    assert enroot_container_name("agentfly-abc-123") == "agentfly-abc-123"


def test_name_without_id_is_random_but_prefixed():
    a, b = enroot_container_name(None), enroot_container_name("")
    assert a.startswith("agentfly-") and b.startswith("agentfly-")
    assert a != b


# ---- owner records and the orphan sweep -------------------------------------

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

from agentfly.resources.containers import enroot_containers as ec  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_owners(tmp_path, monkeypatch):
    """Every test gets its own owner-record folder; no real background sweeps."""
    folder = tmp_path / "owners"
    monkeypatch.setattr(ec, "owners_dir", lambda: folder)
    monkeypatch.setattr(ec, "_sweep_started", True)
    monkeypatch.delenv(ec.ENROOT_SWEEP_ENV, raising=False)
    return folder


@pytest.fixture
def fake_remove(monkeypatch):
    """Replace container removal; ``fail`` lists names that refuse to go."""
    calls = SimpleNamespace(names=[], fail=set())

    def remove(names):
        calls.names.extend(names)
        return [n for n in names if n not in calls.fail]

    monkeypatch.setattr(ec, "_remove_containers", remove)
    return calls


def dead_pid():
    """A PID that is certainly not running: a child we started and reaped."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def write_record(folder, name, **overrides):
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(ec._identity(), container=name)
    record.update(overrides)
    (folder / f"{name}.json").write_text(json.dumps(record))
    return folder / f"{name}.json"


def test_record_and_forget(isolated_owners):
    ec.record_container("agentfly-a")
    record = json.loads((isolated_owners / "agentfly-a.json").read_text())
    assert record["pid"] == os.getpid() and record["container"] == "agentfly-a"
    assert ec.owner_alive(record) is True
    ec.forget_container("agentfly-a")
    assert not (isolated_owners / "agentfly-a.json").exists()


def test_forget_leaves_another_owners_record(isolated_owners):
    path = write_record(isolated_owners, "agentfly-b", pid=dead_pid())
    ec.forget_container("agentfly-b")
    assert path.exists()


def test_owner_alive_judgements():
    me = dict(ec._identity())
    assert ec.owner_alive(me) is True
    assert ec.owner_alive(dict(me, pid=dead_pid())) is False
    assert ec.owner_alive(dict(me, start_time="1")) is False  # a reused PID: same number, other process
    assert ec.owner_alive(dict(me, machine_id="another-machine")) is None
    assert ec.owner_alive(dict(me, pid_namespace="pid:[1]")) is None
    assert ec.owner_alive({"pid": "x"}) is None


def test_sweep_removes_only_dead_owners(isolated_owners, fake_remove):
    write_record(isolated_owners, "agentfly-dead", pid=dead_pid())
    write_record(isolated_owners, "agentfly-reused", start_time="1")
    live = write_record(isolated_owners, "agentfly-live")
    other_machine = write_record(isolated_owners, "agentfly-elsewhere", pid=dead_pid(), machine_id="m2")
    other_ns = write_record(isolated_owners, "agentfly-other-ns", pid=dead_pid(), pid_namespace="pid:[1]")
    assert ec.sweep_orphaned_containers() == ["agentfly-dead", "agentfly-reused"]
    assert sorted(fake_remove.names) == ["agentfly-dead", "agentfly-reused"]
    assert live.exists() and other_machine.exists() and other_ns.exists()
    assert not (isolated_owners / "agentfly-dead.json").exists()


def test_record_kept_when_removal_fails(isolated_owners, fake_remove):
    path = write_record(isolated_owners, "agentfly-stuck", pid=dead_pid())
    fake_remove.fail.add("agentfly-stuck")
    assert ec.sweep_orphaned_containers() == []
    assert path.exists()  # retried by a later sweep


def test_sweep_can_be_switched_off(isolated_owners, fake_remove, monkeypatch):
    write_record(isolated_owners, "agentfly-dead", pid=dead_pid())
    monkeypatch.setenv(ec.ENROOT_SWEEP_ENV, "0")
    assert ec.sweep_orphaned_containers() == [] and fake_remove.names == []


def test_only_one_sweeper_at_a_time(isolated_owners, fake_remove):
    import fcntl

    write_record(isolated_owners, "agentfly-dead", pid=dead_pid())
    with open(isolated_owners / ".sweep.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # another process is sweeping
        assert ec.sweep_orphaned_containers() == [] and fake_remove.names == []
    assert ec.sweep_orphaned_containers() == ["agentfly-dead"]


def test_sweep_never_raises(monkeypatch):
    def boom():
        raise PermissionError("no access")

    monkeypatch.setattr(ec, "owners_dir", boom)
    assert ec.sweep_orphaned_containers() == []


def test_record_never_overwrites_a_live_owner(isolated_owners, fake_remove):
    path = write_record(isolated_owners, "agentfly-shared", pid=1, start_time=ec._process_start_time(1))
    ec.record_container("agentfly-shared")
    assert json.loads(path.read_text())["pid"] == 1 and fake_remove.names == []


def test_record_takes_over_a_dead_owners_name(isolated_owners, fake_remove):
    path = write_record(isolated_owners, "agentfly-reuse", pid=dead_pid())
    ec.record_container("agentfly-reuse")
    assert fake_remove.names == ["agentfly-reuse"]  # its leftover container is removed first
    assert json.loads(path.read_text())["pid"] == os.getpid()


def test_background_sweep_starts_once_per_process(monkeypatch):
    runs = []
    monkeypatch.setattr(ec, "_sweep_started", False)
    monkeypatch.setattr(ec, "sweep_orphaned_containers", lambda: runs.append(1))
    for _ in range(3):
        ec.start_background_sweep()
    import time

    time.sleep(0.2)
    assert runs == [1]


def test_cli_train_runs_the_owner_aware_sweep(monkeypatch):
    """``agentfly train`` sweeps orphans (not every agentfly-* container) before the trainer."""
    import sys
    import types

    import agentfly.cli as cli
    from agentfly.resources import containers

    calls = []
    monkeypatch.setattr(containers, "sweep_orphaned_containers", lambda: calls.append("swept") or [])
    monkeypatch.setattr(containers, "clear_agentfly_containers", lambda: calls.append("cleared-all"))
    stub = types.SimpleNamespace(__file__="main_ppo.py", main=lambda: calls.append("trained"))
    monkeypatch.setattr(cli, "import_module", lambda name, package=None: stub)
    monkeypatch.setattr(sys, "argv", ["agentfly", "train"])
    cli.main()
    assert calls == ["swept", "trained"]


# ---- the local runner really names its container with the prefix ------------

class _FakeContainer:
    def __init__(self, name):
        self.name = name

    async def start_async(self, **kw):
        pass

    def reload(self):
        pass


class _FakeContainers:
    def __init__(self):
        self.create_kwargs = None

    async def create_async(self, image, **kwargs):
        self.create_kwargs = dict(kwargs)
        return _FakeContainer(kwargs["name"])


def test_local_runner_creates_prefixed_enroot_container():
    from agentfly.resources.runner import _start_enroot_container
    from agentfly.resources.types import ContainerResourceSpec

    client = SimpleNamespace(containers=_FakeContainers())
    spec = ContainerResourceSpec(image="ubuntu:22.04", category="container")
    registry = {}
    resource = asyncio.run(
        _start_enroot_container(client, spec, "0f626053-74f7", containers_registry=registry)
    )
    assert client.containers.create_kwargs["name"] == "agentfly-0f626053-74f7"
    assert resource.resource_id == "agentfly-0f626053-74f7"
    assert list(registry) == ["agentfly-0f626053-74f7"]


def test_require_enroot_explains_a_missing_enroot(monkeypatch):
    from agentfly.resources.containers import enroot_containers

    monkeypatch.setattr(enroot_containers.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="needs the `enroot` command.*AF_CONTAINER_ENGINE"):
        enroot_containers.require_enroot()

    monkeypatch.setattr(enroot_containers.shutil, "which", lambda _: "/usr/bin/enroot")
    enroot_containers.require_enroot()  # no error when enroot is installed


def test_local_runner_records_the_owner_before_creating(isolated_owners):
    from agentfly.resources.runner import _start_enroot_container
    from agentfly.resources.types import ContainerResourceSpec

    seen = {}

    class RecordingContainers(_FakeContainers):
        async def create_async(self, image, **kwargs):
            seen["record_existed"] = (isolated_owners / f"{kwargs['name']}.json").exists()
            return await super().create_async(image, **kwargs)

    client = SimpleNamespace(containers=RecordingContainers())
    asyncio.run(_start_enroot_container(client, ContainerResourceSpec(image="x", category="container"), "rid", containers_registry={}))
    assert seen["record_existed"] is True
