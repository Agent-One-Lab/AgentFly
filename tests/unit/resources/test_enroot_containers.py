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
    ENROOT_CLEANUP_ENV,
    ENROOT_CONTAINER_PREFIX,
    clear_agentfly_containers,
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


# ---- pre-training sweep -------------------------------------------------------

@pytest.fixture
def fake_clear(monkeypatch):
    """Replace ``enroot.clear_enroot_containers`` and record how the sweep calls it."""
    import enroot

    calls = []

    def clear_enroot_containers(prefix=None):
        calls.append(prefix)
        return ["agentfly-a"]

    monkeypatch.setattr(enroot, "clear_enroot_containers", clear_enroot_containers)
    return calls


def test_sweep_calls_enroot_with_the_agentfly_prefix(monkeypatch, fake_clear):
    monkeypatch.delenv(ENROOT_CLEANUP_ENV, raising=False)
    assert clear_agentfly_containers() == ["agentfly-a"]
    assert fake_clear == ["agentfly-"], "must never sweep containers outside the agentfly- prefix"


def test_sweep_can_be_switched_off(monkeypatch, fake_clear):
    monkeypatch.setenv(ENROOT_CLEANUP_ENV, "0")
    assert clear_agentfly_containers() == []
    assert fake_clear == []


def test_sweep_never_raises(monkeypatch, caplog):
    import enroot

    monkeypatch.delenv(ENROOT_CLEANUP_ENV, raising=False)

    def boom(prefix=None):
        raise FileNotFoundError("enroot: command not found")

    monkeypatch.setattr(enroot, "clear_enroot_containers", boom)
    with caplog.at_level("WARNING"):
        assert clear_agentfly_containers() == []  # must not propagate
    assert "enroot cleanup skipped" in caplog.text


def test_cli_train_runs_the_sweep(monkeypatch):
    """``agentfly train`` sweeps before importing the trainer (which we stub out)."""
    import sys
    import types

    import agentfly.cli as cli
    from agentfly.resources import containers

    calls = []
    monkeypatch.setattr(containers, "clear_agentfly_containers", lambda: calls.append("swept"))
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
