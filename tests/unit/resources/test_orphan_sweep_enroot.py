"""Orphaned-container cleanup with real enroot containers and real owner processes.

Each test starts owner processes that create a container through AgentFly, ends them in
different ways, and checks which containers the sweep removes. Every test uses its own
``AF_CACHE_DIR`` for owner records and unique container names, and removes leftovers.
"""

import os
import signal
import socket
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

pytest.importorskip("enroot")
if subprocess.run(["which", "enroot"], capture_output=True).returncode != 0:
    pytest.skip("enroot is not installed", allow_module_level=True)

from agentfly.resources.containers import enroot_containers as ec  # noqa: E402

IMAGE = "python:3.12"

OWNER = textwrap.dedent(
    """
    import asyncio, sys, time
    from agentfly.core import Context
    from agentfly.resources import ContainerResourceSpec, ResourceEngine

    name, mode = sys.argv[1], sys.argv[2]

    async def main():
        ctx = Context(rollout_id=name)
        spec = ContainerResourceSpec(category="container", image="%s")
        await ctx.acquire_resource(spec=spec, id=name, scope="global")
        await ctx.release_resource(scope="global")      # back to the pool, still alive
        if mode == "reuse":                             # a second run reuses the pooled container
            again = await ctx.acquire_resource(spec=spec, id=name, scope="global")
            print("REUSED", again.resource_id, flush=True)
        if mode == "close":
            await ResourceEngine.close()
        print("READY", flush=True)
        if mode == "raise":
            raise RuntimeError("crash after creating a container")
        if mode == "sleep":
            time.sleep(600)

    asyncio.run(main())
    """
    % IMAGE
)


def containers():
    out = subprocess.run(["enroot", "list"], capture_output=True, text=True).stdout
    return set(out.split())


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A private AF_CACHE_DIR shared by the test process and its owner processes."""
    cache = tmp_path / "cache"
    folder = cache / "enroot" / "owners" / f"{socket.gethostname()}-{ec._machine_id()[:12]}"
    monkeypatch.setattr(ec, "owners_dir", lambda: folder)
    monkeypatch.delenv(ec.ENROOT_SWEEP_ENV, raising=False)
    names = []
    environment = dict(os.environ, AF_CACHE_DIR=str(cache), HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    environment.pop(ec.ENROOT_SWEEP_ENV, None)
    yield SimpleEnv(folder, environment, names)
    for name in names:  # leave nothing behind, whatever happened
        subprocess.run(["enroot", "remove", "--force", name], capture_output=True)


class SimpleEnv:
    def __init__(self, folder, environment, names):
        self.folder, self.environment, self.names = folder, environment, names

    def name(self):
        name = f"agentfly-sweeptest-{uuid.uuid4().hex[:8]}"
        self.names.append(name)
        return name

    def owner(self, name, mode, **extra_env):
        """Start an owner process; return it once its container exists."""
        proc = subprocess.Popen(
            [sys.executable, "-c", OWNER, name, mode],
            env=dict(self.environment, **extra_env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        lines = []
        for line in proc.stdout:
            lines.append(line)
            if line.startswith("READY"):
                break
        assert any(x.startswith("READY") for x in lines), proc.stderr.read()[-2000:]
        proc.lines = lines
        return proc

    def record(self, name) -> Path:
        return self.folder / f"{name}.json"


def test_killed_owner_is_cleaned_up_by_the_next_process(env):
    orphan, survivor = env.name(), env.name()
    owner = env.owner(orphan, "sleep")
    owner.send_signal(signal.SIGKILL)
    owner.wait()
    assert orphan in containers() and env.record(orphan).exists()  # left behind by kill -9

    # The next AgentFly process sweeps in the background when it creates its first container.
    nxt = env.owner(survivor, "sleep")
    try:
        for _ in range(60):
            if orphan not in containers():
                break
            subprocess.run(["sleep", "1"])
        assert orphan not in containers() and not env.record(orphan).exists()
        assert survivor in containers()  # the live process's container is untouched
    finally:
        nxt.terminate()
        nxt.wait()


@pytest.mark.parametrize("mode", ["exit", "raise"])
def test_owner_that_exits_without_cleanup_is_swept(env, mode):
    name = env.name()
    owner = env.owner(name, mode)
    owner.wait()
    assert name in containers()
    assert ec.sweep_orphaned_containers() == [name]
    assert name not in containers() and not env.record(name).exists()


def test_live_owner_is_never_swept(env):
    name = env.name()
    owner = env.owner(name, "sleep")
    try:
        assert ec.sweep_orphaned_containers() == []
        assert name in containers() and env.record(name).exists()
    finally:
        owner.send_signal(signal.SIGTERM)  # a Slurm-style kill
        owner.wait()
    assert ec.sweep_orphaned_containers() == [name]
    assert name not in containers()


def test_proper_close_leaves_no_record_or_container(env):
    name = env.name()
    env.owner(name, "close").wait()
    assert name not in containers() and not env.record(name).exists()
    assert ec.sweep_orphaned_containers() == []


def test_repeated_runs_reuse_one_container_and_one_record(env):
    name = env.name()
    owner = env.owner(name, "reuse")
    owner.wait()
    assert any(line.strip() == f"REUSED {name}" for line in owner.lines)
    assert [p.name for p in env.folder.glob("*.json")] == [f"{name}.json"]
    assert ec.sweep_orphaned_containers() == [name]


def test_unrecorded_containers_are_never_touched(env):
    # A container made outside AgentFly's resource engine (e.g. with enroot directly) has no record.
    name = env.name()
    image = Path(os.environ.get("ENROOT_IMAGES_PATH", Path.home() / ".cache/enroot/images")) / "python+3.12.sqsh"
    if not image.exists():
        pytest.skip("python:3.12 image not cached")
    subprocess.run(["enroot", "create", "--name", name, str(image)], capture_output=True, check=True)
    env.folder.mkdir(parents=True, exist_ok=True)
    assert ec.sweep_orphaned_containers() == []
    assert name in containers()


def test_switching_the_sweep_off_keeps_orphans(env, monkeypatch):
    orphan, other = env.name(), env.name()
    env.owner(orphan, "exit").wait()
    nxt = env.owner(other, "sleep", AF_ENROOT_SWEEP="0")
    try:
        subprocess.run(["sleep", "3"])
        assert orphan in containers()  # the background sweep was switched off
    finally:
        nxt.terminate()
        nxt.wait()
    monkeypatch.setenv(ec.ENROOT_SWEEP_ENV, "0")
    assert ec.sweep_orphaned_containers() == []
