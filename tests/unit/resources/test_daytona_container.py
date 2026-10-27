"""Tests for the Daytona container engine.

Three layers:

  * Pure-logic tests (always run, no SDK calls): resource clamping, memory
    parsing, snapshot naming, and the content digest that decides whether a
    Dockerfile reuses a snapshot or builds a new one.

  * Fake-sandbox tests (no credentials): a stand-in object with the Daytona
    SDK's surface (``process.exec``, ``fs.upload_file``, ``refresh_data``,
    ``delete``) driven through the REAL ``DaytonaContainer`` and the inherited
    ``ContainerResource`` lifecycle — this is what actually exercises the
    handle contract that ``run_cmd`` depends on.

  * A live round-trip, skipped unless DAYTONA_API_KEY is set (and opted into
    with RUN_DAYTONA_LIVE=1, since it provisions a real cloud sandbox).
"""

from __future__ import annotations

import asyncio
import os
import tarfile
import io
from pathlib import Path

import pytest

pytest.importorskip("daytona")

from agentfly.resources.types import ContainerResourceSpec, DaytonaResourceSpec, ResourceStatus
from agentfly.resources.runner import _resolve_container_engine
from agentfly.resources.containers import daytona_container as dc


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("4g", 4), ("4G", 4), ("4gb", 4),
    ("512m", 1),          # sub-GiB still has to ask for a whole GiB
    ("2048m", 2),
    ("1048576k", 1),
    (8 * 1024 ** 3, 8),   # raw byte count
    (None, None),
    ("nonsense", None),
])
def test_parse_mem_gb(value, expected):
    assert dc._parse_mem_gb(value) == expected


def test_resources_clamped_to_account_limits(monkeypatch):
    """Over-large asks are clamped, not passed through — Daytona rejects them
    outright, so an unclamped spec would fail sandbox creation."""
    spec = ContainerResourceSpec(category="container", cpu_count=16,
                                 mem_limit="64g", disk_gb=100)
    res = dc._resources_for(spec)
    assert res.cpu == dc._DEFAULT_DAYTONA.max_cpus
    assert res.memory == dc._DEFAULT_DAYTONA.max_memory_gb
    assert res.disk == dc._DEFAULT_DAYTONA.max_disk_gb


def test_resources_none_when_unspecified():
    spec = ContainerResourceSpec(category="container")
    assert dc._resources_for(spec) is None


def test_context_digest_is_stable_and_content_sensitive(tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "a.txt").write_text("hello")
    df = "FROM ubuntu:22.04\nCOPY a.txt /a.txt\n"

    d1 = dc._context_digest(df, str(ctx))
    d2 = dc._context_digest(df, str(ctx))
    assert d1 == d2, "same inputs must reuse the same snapshot"

    (ctx / "a.txt").write_text("goodbye")
    assert dc._context_digest(df, str(ctx)) != d1, "context change must rebuild"

    assert dc._context_digest(df + "RUN true\n", str(ctx)) != d1, \
        "dockerfile change must rebuild"


def test_context_digest_ignores_pycache(tmp_path):
    ctx = tmp_path / "ctx"
    (ctx / "__pycache__").mkdir(parents=True)
    (ctx / "keep.txt").write_text("x")
    before = dc._context_digest("FROM x\n", str(ctx))
    (ctx / "__pycache__" / "junk.pyc").write_bytes(b"\x00\x01")
    assert dc._context_digest("FROM x\n", str(ctx)) == before


def test_snapshot_name_is_target_scoped(monkeypatch):
    monkeypatch.delenv("DAYTONA_TARGET", raising=False)
    assert dc._snapshot_name("abc123") == "agentfly-abc123"
    monkeypatch.setenv("DAYTONA_TARGET", "EU")
    assert dc._snapshot_name("abc123") == "agentfly-abc123-eu"


def test_image_from_materializes_dockerfile_and_context(tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "seed.txt").write_text("payload")
    _image, ctx_dir = dc._image_from("FROM ubuntu:22.04\nCOPY seed.txt /s\n", str(ctx))
    try:
        assert (Path(ctx_dir) / "Dockerfile").read_text().startswith("FROM ubuntu")
        assert (Path(ctx_dir) / "seed.txt").read_text() == "payload"
    finally:
        import shutil
        shutil.rmtree(ctx_dir, ignore_errors=True)


def test_engine_resolution_picks_daytona(monkeypatch):
    spec = ContainerResourceSpec(category="container", container_engine="daytona")
    assert _resolve_container_engine(spec) == "daytona"
    monkeypatch.setenv("AF_CONTAINER_ENGINE", "daytona")
    assert _resolve_container_engine(
        ContainerResourceSpec(category="container")) == "daytona"


# ---------------------------------------------------------------------------
# Fake sandbox — drives the real handle + ContainerResource lifecycle
# ---------------------------------------------------------------------------

class _FakeExec:
    def __init__(self, exit_code=0, result=""):
        self.exit_code = exit_code
        self.result = result
        self.artifacts = None


class _FakeProcess:
    def __init__(self, owner):
        self.owner = owner

    def exec(self, command, cwd=None, env=None, timeout=None):
        self.owner.commands.append({"command": command, "cwd": cwd,
                                    "env": env, "timeout": timeout})
        if self.owner.raise_timeout:
            raise RuntimeError("operation timed out")
        return _FakeExec(self.owner.exit_code, self.owner.output)


class _FakeFs:
    def __init__(self, owner):
        self.owner = owner

    def upload_file(self, data, remote_path, timeout=None):
        self.owner.uploads.append((remote_path, data))


class _FakeSandbox:
    id = "sandbox-123"

    def __init__(self, state="started"):
        self.state = state
        self.commands = []
        self.uploads = []
        self.deleted = False
        self.exit_code = 0
        self.output = ""
        self.raise_timeout = False
        self.process = _FakeProcess(self)
        self.fs = _FakeFs(self)

    def refresh_data(self):
        pass

    def delete(self):
        self.deleted = True


def _container(state="started", **kw):
    sb = _FakeSandbox(state=state)
    for k, v in kw.items():
        setattr(sb, k, v)
    return dc.DaytonaContainer("c1", sandbox=sb), sb


@pytest.mark.parametrize("state,expected", [
    ("started", ResourceStatus.RUNNING),
    ("running", ResourceStatus.RUNNING),
    ("stopped", ResourceStatus.STOPPED),
    ("destroyed", ResourceStatus.STOPPED),
    ("error", ResourceStatus.STOPPED),
    ("creating", ResourceStatus.PENDING),
])
def test_status_mapping(state, expected):
    c, _ = _container(state=state)
    assert asyncio.run(c.get_status()) is expected


def test_exec_merges_stderr_and_returns_exec_result():
    """docker's exec_run folds stderr into stdout; parity matters because
    callers scrape the combined stream (e.g. the `__EXIT__:<n>` grader marker)."""
    c, sb = _container(output="hi", exit_code=3)
    res = c.exec_run(["bash", "-c", "echo hi"])
    assert res.exit_code == 3
    assert res.output == b"hi"
    assert sb.commands[-1]["command"].endswith(" 2>&1")
    # ExecResult tuple-compat, like DockerExecResult
    code, out = res
    assert (code, out) == (3, b"hi")


def test_exec_drops_privileges_when_user_given():
    c, sb = _container()
    c.exec_run("whoami", user="agent")
    assert sb.commands[-1]["command"].startswith("su agent -s /bin/bash -c ")


def test_exec_passes_cwd_and_env():
    c, sb = _container()
    c.exec_run("ls", workdir="/workspace", environment={"A": "1"})
    assert sb.commands[-1]["cwd"] == "/workspace"
    assert sb.commands[-1]["env"] == {"A": "1"}


def test_exec_falls_back_to_container_workdir():
    sb = _FakeSandbox()
    c = dc.DaytonaContainer("c1", sandbox=sb, workdir="/workspace")
    c.exec_run("ls")
    assert sb.commands[-1]["cwd"] == "/workspace"


def test_exec_timeout_maps_to_asyncio_timeout():
    """ContainerResource.run_cmd only recognizes asyncio.TimeoutError, so an
    SDK timeout must be translated or it surfaces as a generic failure."""
    c, _ = _container(raise_timeout=True)
    with pytest.raises(asyncio.TimeoutError):
        c.exec_run("sleep 100", timeout=1)


def test_run_cmd_roundtrip_through_container_resource():
    """The inherited run_cmd base64-wraps the command and drives the handle —
    this is the path every tool call actually takes."""
    c, sb = _container(output="done")
    out = asyncio.run(c.run_cmd("echo done"))
    assert out == "done"
    sent = sb.commands[-1]["command"]
    assert "base64 -d | bash -s" in sent


def test_put_archive_uploads_tar_and_unpacks():
    c, sb = _container()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("f.txt")
        data = b"content"
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    assert c.put_archive("/dest", buf.getvalue()) is True
    remote_tar, payload = sb.uploads[-1]
    assert remote_tar.startswith("/tmp/agentfly-") and remote_tar.endswith(".tar")
    assert payload == buf.getvalue()
    cmd = sb.commands[-1]["command"]
    assert "mkdir -p /dest" in cmd and "tar -xf" in cmd and "rm -f" in cmd


def test_put_archive_reports_failure():
    c, _ = _container(exit_code=1)
    assert c.put_archive("/dest", b"junk") is False


def test_copy_to_dir_copies_contents(tmp_path):
    src = tmp_path / "d"
    src.mkdir()
    (src / "a.txt").write_text("A")
    c, sb = _container()
    c.copy_to(str(src), "/target")
    _, payload = sb.uploads[-1]
    with tarfile.open(fileobj=io.BytesIO(payload)) as tf:
        names = {n.lstrip("./") for n in tf.getnames()}
    assert "a.txt" in names, "directory CONTENTS, not the directory itself"


def test_copy_in_delegates_to_copy_to_async(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")
    c, sb = _container()
    asyncio.run(c.copy_in(str(f), "/target"))
    assert sb.uploads, "copy_in must reach the sandbox"


def test_kill_deletes_sandbox_once_and_is_idempotent():
    c, sb = _container()
    asyncio.run(c.end())
    assert sb.deleted is True
    sb.deleted = False
    asyncio.run(c.close())          # second teardown must be a no-op
    assert sb.deleted is False
    assert asyncio.run(c.get_status()) is ResourceStatus.STOPPED


def test_kill_never_raises():
    """Teardown runs in finally blocks; an exception there would mask the real
    error and leak the resource id."""
    c, sb = _container()

    def boom():
        raise RuntimeError("network down")

    sb.delete = boom
    asyncio.run(c.end())            # must not raise


def test_unreachable_sandbox_reads_as_exited():
    c, sb = _container()

    def boom():
        raise RuntimeError("404 not found")

    sb.refresh_data = boom
    assert asyncio.run(c.get_status()) is ResourceStatus.STOPPED


def test_resource_identity():
    c, _ = _container()
    assert c.resource_id == "c1"
    assert c.category == "container"
    assert c.sandbox_id == "sandbox-123"


# ---------------------------------------------------------------------------
# Snapshot policy / fallback
# ---------------------------------------------------------------------------

class _DeniedClient:
    """Client whose snapshot writes are refused, as a read-only API key's are."""

    def __init__(self):
        self.created = []

        class _Snap:
            def get(_self, name):
                raise RuntimeError("404 not found")

            def create(_self, params, on_logs=None, timeout=None):
                raise RuntimeError("Failed to create snapshot: Access denied")

        self.snapshot = _Snap()

    def create(self, params, timeout=None, on_snapshot_create_logs=None):
        self.created.append(params)
        return _FakeSandbox()


def test_snapshot_denial_falls_back_to_declarative_build(monkeypatch, tmp_path):
    """A key without snapshot write scope must still be able to run: the engine
    degrades to a per-sandbox declarative build instead of failing the rollout."""
    from daytona import CreateSandboxFromImageParams

    client = _DeniedClient()
    monkeypatch.setattr(dc, "daytona_client", lambda: client)
    monkeypatch.setattr(dc, "_snapshots_denied", False)

    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "seed.txt").write_text("payload")
    spec = ContainerResourceSpec(
        category="container", container_engine="daytona",
        dockerfile="FROM ubuntu:22.04\nCOPY seed.txt /seed.txt\n",
        build_context=str(ctx),
        daytona=DaytonaResourceSpec(snapshots="auto"))

    c = dc.start_daytona_container(name="c1", spec=spec)
    assert isinstance(client.created[-1], CreateSandboxFromImageParams)
    assert dc._snapshots_denied is True, "denial must be cached, not re-probed"
    assert c.name == "c1"


def test_snapshot_mode_on_propagates_denial(monkeypatch, tmp_path):
    """With snapshots explicitly required, a denial is an error — silently
    rebuilding per sandbox would be a large, invisible cost regression."""
    client = _DeniedClient()
    monkeypatch.setattr(dc, "daytona_client", lambda: client)
    monkeypatch.setattr(dc, "_snapshots_denied", False)

    spec = ContainerResourceSpec(
        category="container", dockerfile="FROM ubuntu:22.04\n",
        daytona=DaytonaResourceSpec(snapshots="on"))
    with pytest.raises(dc.SnapshotUnavailable):
        dc.start_daytona_container(name="c1", spec=spec)


def test_build_failure_is_not_treated_as_denial(monkeypatch):
    """A broken Dockerfile must surface, not silently degrade."""
    client = _DeniedClient()

    def boom(params, on_logs=None, timeout=None):
        raise RuntimeError("failed to solve: dockerfile parse error")

    client.snapshot.create = boom
    monkeypatch.setattr(dc, "daytona_client", lambda: client)
    with pytest.raises(RuntimeError, match="snapshot build failed"):
        dc.ensure_snapshot("FROM nope\n", None)


def test_declarative_build_cleans_up_context_dir(monkeypatch, tmp_path):
    client = _DeniedClient()
    monkeypatch.setattr(dc, "daytona_client", lambda: client)
    seen = {}
    real = dc._image_from

    def spy(dockerfile, build_context):
        img, ctx = real(dockerfile, build_context)
        seen["ctx"] = ctx
        return img, ctx

    monkeypatch.setattr(dc, "_image_from", spy)
    dc.start_daytona_container(
        name="c1",
        spec=ContainerResourceSpec(category="container",
                                   dockerfile="FROM ubuntu:22.04\n",
                                   daytona=DaytonaResourceSpec(snapshots="off")))
    assert not Path(seen["ctx"]).exists(), "temp build context must not leak"


# ---------------------------------------------------------------------------
# Live round-trip (opt-in)
# ---------------------------------------------------------------------------

live = pytest.mark.skipif(
    not (os.environ.get("DAYTONA_API_KEY") and os.environ.get("RUN_DAYTONA_LIVE")),
    reason="needs DAYTONA_API_KEY and RUN_DAYTONA_LIVE=1 (provisions a real sandbox)",
)


@live
def test_live_dockerfile_roundtrip(tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "seed.txt").write_text("payload\n")
    spec = ContainerResourceSpec(
        category="container",
        container_engine="daytona",
        dockerfile="FROM ubuntu:22.04\nCOPY seed.txt /seed.txt\nWORKDIR /workspace\n",
        build_context=str(ctx),
        cpu_count=1,
        mem_limit="2g",
    )
    c = dc.start_daytona_container(name=None, workdir="/workspace", spec=spec)
    try:
        assert asyncio.run(c.get_status()) is ResourceStatus.RUNNING
        assert asyncio.run(c.run_cmd("cat /seed.txt")).strip() == "payload"
        assert asyncio.run(c.run_cmd("pwd")).strip() == "/workspace"
        local = tmp_path / "up.txt"
        local.write_text("uploaded")
        asyncio.run(c.copy_in(str(local), "/workspace/sub"))
        assert "uploaded" in asyncio.run(c.run_cmd("cat /workspace/sub/up.txt"))
    finally:
        asyncio.run(c.end())
