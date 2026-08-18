"""File tools (create/read/edit/list/run_python) on the DOCKER engine.

Mirrors the sibling file-tool tests but forces ``container_engine=docker``.
This exercises two things at once:

  * the file tools running unchanged on the docker backend, and
  * the docker backend's bind-mount staging — the file tools mount
    ``file_manager.py`` into the container, and when the daemon is remote
    (``DOCKER_HOST=ssh://...``) that source is rsynced to the remote host
    first. So these tests are the real check that mounts work over SSH.

The whole module skips when no docker daemon is reachable. Point it at a
remote daemon with ``DOCKER_HOST=ssh://user@host``. Image is a generic
python image (has ``python3``, which file_manager needs); override with
``TEST_DOCKER_IMAGE``.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from agentfly.core import Context
from agentfly.tools.src.file.tools import (
    create_file,
    edit_file,
    list_files,
    read_file,
    run_python,
)


IMAGE_ID = os.environ.get("TEST_DOCKER_IMAGE", "python:3.11-slim")


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


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="docker daemon not reachable (set DOCKER_HOST=ssh://...)",
)


@pytest.fixture(autouse=True)
def _force_docker_engine(monkeypatch):
    """Make the file tools' container spec resolve to the docker engine."""
    monkeypatch.setenv("AF_CONTAINER_ENGINE", "docker")


def _ctx(rid: str) -> Context:
    return Context(
        rollout_id=rid,
        group_id="file_tools_docker",
        metadata={"image_id": IMAGE_ID},
    )


def _strip_line_numbers(content: str) -> str:
    """Remove '  N | ' prefix from read_file output."""
    out = []
    for line in content.split("\n"):
        if "|" in line:
            _, _, rest = line.partition("|")
            out.append(rest.strip())
        else:
            out.append(line)
    return "\n".join(out)


@pytest.mark.asyncio
async def test_create_read_edit_list_run_on_docker():
    """Full file-tool lifecycle inside a docker container."""
    ctx = _ctx("ft_docker_main")
    try:
        # create_file makedirs the workspace root, so a fresh image is fine.
        r = await create_file(
            path="hello.py",
            content="print('hi from docker')\nX = 1\n",
            context=ctx,
        )
        assert "Error" not in r["observation"], r["observation"]

        # read_file sees the content
        r = await read_file(path="hello.py", context=ctx)
        body = _strip_line_numbers(r["observation"])
        assert "hi from docker" in body, r["observation"]

        # list_files sees the new file
        r = await list_files(path=".", context=ctx)
        assert "hello.py" in r["observation"], r["observation"]

        # edit_file replaces a block
        r = await edit_file(
            path="hello.py",
            search_block="X = 1",
            replace_block="X = 2  # edited by docker test",
            context=ctx,
        )
        assert "Error" not in r["observation"], r["observation"]
        r = await read_file(path="hello.py", context=ctx)
        assert "X = 2" in _strip_line_numbers(r["observation"]), r["observation"]

        # run_python executes the script (proves python3 + workspace cwd work)
        r = await run_python(path="hello.py", context=ctx)
        assert "hi from docker" in r["observation"], r["observation"]
    finally:
        await ctx.end_resource(scope="rollout")


@pytest.mark.asyncio
async def test_edit_missing_block_errors_on_docker():
    """A non-existent search block returns an error, not a silent pass."""
    ctx = _ctx("ft_docker_missing")
    try:
        await create_file(path="a.txt", content="line one\n", context=ctx)
        r = await edit_file(
            path="a.txt",
            search_block="__SEARCH_BLOCK_DOES_NOT_EXIST__",
            replace_block="replacement",
            context=ctx,
        )
        obs = r["observation"]
        assert "Error" in obs, obs
        assert "not found" in obs.lower() or "exact" in obs.lower(), obs
    finally:
        await ctx.end_resource(scope="rollout")


@pytest.mark.asyncio
async def test_read_nonexistent_returns_error_on_docker():
    ctx = _ctx("ft_docker_read_missing")
    try:
        r = await read_file(path="does_not_exist.py", context=ctx)
        # file tools surface a readable error string rather than raising
        assert "Error" in r["observation"] or "No such" in r["observation"], \
            r["observation"]
    finally:
        await ctx.end_resource(scope="rollout")
