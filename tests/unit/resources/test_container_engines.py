"""Tests for the switchable container engine (enroot | docker).

Two layers:

  * Pure-logic tests (always run, no container backend needed): engine
    resolution, docker mount/option translation, and that
    ``ContainerResource`` is engine-neutral (drives a duck-typed handle).

  * Real round-trip tests, parametrized over whichever engines are
    available on this machine (enroot importable / docker daemon
    reachable). Each starts a container via ``LocalRunner`` with the given
    engine, runs commands, checks status, and tears it down — asserting
    identical behaviour across engines.

Select the test image with TEST_CONTAINER_IMAGE (default ubuntu:22.04).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import uuid

import pytest

from agentfly.resources.types import ContainerResourceSpec, ResourceStatus
from agentfly.resources.runner import LocalRunner, _resolve_container_engine
from agentfly.resources.containers.container_resource import ContainerResource
from agentfly.resources.containers.docker_container import (
    DockerExecResult,
    _build_mount_args,
    _ssh_target_from_docker_host,
    _translate_mount_opts,
)


TEST_IMAGE = os.environ.get("TEST_CONTAINER_IMAGE", "ubuntu:22.04")


# ---------------------------------------------------------------------------
# Availability detection -> parametrize real round-trips over present engines
# ---------------------------------------------------------------------------


def _enroot_available() -> bool:
    try:
        import enroot  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


_AVAILABLE_ENGINES = []
if _enroot_available():
    _AVAILABLE_ENGINES.append("enroot")
if _docker_available():
    _AVAILABLE_ENGINES.append("docker")


# ---------------------------------------------------------------------------
# Pure-logic: engine resolution
# ---------------------------------------------------------------------------


def test_engine_defaults_to_enroot(monkeypatch):
    monkeypatch.delenv("AF_CONTAINER_ENGINE", raising=False)
    spec = ContainerResourceSpec(category="container", image="img")
    assert _resolve_container_engine(spec) == "enroot"


def test_engine_env_override(monkeypatch):
    monkeypatch.setenv("AF_CONTAINER_ENGINE", "docker")
    spec = ContainerResourceSpec(category="container", image="img")
    assert _resolve_container_engine(spec) == "docker"


def test_engine_spec_beats_env(monkeypatch):
    monkeypatch.setenv("AF_CONTAINER_ENGINE", "docker")
    spec = ContainerResourceSpec(
        category="container", image="img", container_engine="enroot")
    assert _resolve_container_engine(spec) == "enroot"


def test_engine_case_insensitive(monkeypatch):
    monkeypatch.delenv("AF_CONTAINER_ENGINE", raising=False)
    spec = ContainerResourceSpec(
        category="container", image="img", container_engine="DOCKER")
    assert _resolve_container_engine(spec) == "docker"


# ---------------------------------------------------------------------------
# Pure-logic: docker mount/option translation
# ---------------------------------------------------------------------------


def test_translate_mount_opts():
    assert _translate_mount_opts("ro,rbind") == "ro"
    assert _translate_mount_opts("rw,rbind") == "rw"
    assert _translate_mount_opts("") == "rw"
    assert _translate_mount_opts("rbind") == "rw"


def test_ssh_target_parsing(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "ssh://ubuntu@10.0.0.1")
    assert _ssh_target_from_docker_host() == "ubuntu@10.0.0.1"
    monkeypatch.setenv("DOCKER_HOST", "ssh://user@host:2222/x")
    assert _ssh_target_from_docker_host() == "user@host:2222"
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    assert _ssh_target_from_docker_host() is None
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    assert _ssh_target_from_docker_host() is None


def test_build_mount_args_local(monkeypatch):
    """With a local daemon, mounts are passed straight through (no staging)."""
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    args = _build_mount_args({"/tmp/x": "/usr/local/bin:ro,rbind"}, "cN")
    assert args == ["-v", "/tmp/x:/usr/local/bin:ro"]
    # default rw when no opts
    args = _build_mount_args({"/tmp/y": "/data"}, "cN")
    assert args == ["-v", "/tmp/y:/data:rw"]
    assert _build_mount_args({}, "cN") == []


# ---------------------------------------------------------------------------
# Pure-logic: ContainerResource is engine-neutral (duck-typed handle)
# ---------------------------------------------------------------------------


class _FakeHandle:
    """Minimal stand-in exposing the interface ContainerResource needs."""

    def __init__(self):
        self.name = "fake-c"
        self.status = "running"
        self.killed = False
        self.last_argv = None

    def reload(self):
        pass

    async def exec_run_async(self, argv, **kwargs):
        self.last_argv = argv
        return DockerExecResult(exit_code=0, output=b"ok-from-handle\n")

    async def kill_async(self, **kwargs):
        self.killed = True


@pytest.mark.asyncio
async def test_container_resource_runs_over_duck_handle():
    handle = _FakeHandle()
    spec = ContainerResourceSpec(category="container", image="img")
    res = ContainerResource(handle, "fake-c", spec)

    out = await res.run_cmd("echo hi")
    assert out.strip() == "ok-from-handle"
    # run_cmd wraps the command as base64 | base64 -d | bash -s, invoked
    # through bash -c — engine-agnostic, no enroot/docker specifics leak in.
    assert handle.last_argv[:2] == ["bash", "-c"]
    assert "base64 -d" in handle.last_argv[2]

    status = await res.get_status()
    assert status == ResourceStatus.RUNNING

    await res.end()
    assert handle.killed is True


@pytest.mark.asyncio
async def test_container_resource_run_cmd_timeout_maps_to_asyncio():
    """A handle raising on timeout surfaces as asyncio.TimeoutError."""
    class _TimeoutHandle(_FakeHandle):
        async def exec_run_async(self, argv, **kwargs):
            raise asyncio.TimeoutError("backend timed out")

    res = ContainerResource(_TimeoutHandle(), "c",
                            ContainerResourceSpec(category="container", image="img"))
    with pytest.raises(asyncio.TimeoutError):
        await res.run_cmd("sleep 100", timeout=1)


# ---------------------------------------------------------------------------
# Real round-trips, parametrized over available engines
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _AVAILABLE_ENGINES,
                    reason="no container engine available (enroot/docker)")
@pytest.mark.parametrize("engine", _AVAILABLE_ENGINES)
@pytest.mark.asyncio
async def test_engine_start_exec_end(engine):
    """Start a container with the given engine, run commands, tear down."""
    runner = LocalRunner()
    rid = f"engtest-{engine}-{uuid.uuid4().hex[:10]}"
    spec = ContainerResourceSpec(
        category="container", image=TEST_IMAGE, container_engine=engine)

    resource = await runner.start_resource(spec, resource_id=rid, timeout=600)
    try:
        assert resource.category == "container"
        assert await runner.get_status(resource) == ResourceStatus.RUNNING

        out = await resource.run_cmd("echo hello-$((1+1))")
        assert out.strip() == "hello-2"

        # stdout+stderr are combined, exactly like the enroot path
        out2 = await resource.run_cmd("echo to-err 1>&2")
        assert "to-err" in out2

        # exit status / failing command still returns output (no raise)
        out3 = await resource.run_cmd("ls /definitely-not-here 2>&1 || true")
        assert "definitely-not-here" in out3
    finally:
        await runner.end_resource(resource)
    assert resource.resource_id not in runner._containers


@pytest.mark.skipif(not _AVAILABLE_ENGINES,
                    reason="no container engine available (enroot/docker)")
@pytest.mark.parametrize("engine", _AVAILABLE_ENGINES)
@pytest.mark.asyncio
async def test_engine_run_cmd_timeout(engine):
    """run_cmd enforces timeout -> asyncio.TimeoutError, on either engine."""
    runner = LocalRunner()
    rid = f"engtmo-{engine}-{uuid.uuid4().hex[:10]}"
    spec = ContainerResourceSpec(
        category="container", image=TEST_IMAGE, container_engine=engine)
    resource = await runner.start_resource(spec, resource_id=rid, timeout=600)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await resource.run_cmd("sleep 30", timeout=2)
    finally:
        await runner.end_resource(resource)


@pytest.mark.skipif(len(_AVAILABLE_ENGINES) < 2,
                    reason="need both enroot and docker to compare engines")
@pytest.mark.asyncio
async def test_engines_agree_on_output():
    """The same command produces the same stdout regardless of engine."""
    runner = LocalRunner()
    cmd = "python3 -c \"print(sum(range(10)))\" 2>/dev/null || echo 45"
    results = {}
    for engine in _AVAILABLE_ENGINES:
        rid = f"engcmp-{engine}-{uuid.uuid4().hex[:8]}"
        spec = ContainerResourceSpec(
            category="container", image=TEST_IMAGE, container_engine=engine)
        resource = await runner.start_resource(spec, resource_id=rid, timeout=600)
        try:
            results[engine] = (await resource.run_cmd(cmd)).strip()
        finally:
            await runner.end_resource(resource)
    assert len(set(results.values())) == 1, f"engine outputs differ: {results}"
