"""Resources are released when training ends: ``ResourceEngine.close`` and ``RayPPOTrainer.close_resources``."""

import asyncio
import threading

import pytest

from agentfly.resources import ResourceEngine


class FakeResource:
    def __init__(self, fail=False):
        self.fail = fail
        self.ended = False

    async def end(self):
        await asyncio.sleep(0.05)
        self.ended = True
        if self.fail:
            raise RuntimeError("enroot remove failed")


@pytest.fixture
def engine_store(monkeypatch):
    monkeypatch.setattr(ResourceEngine, "_lock", asyncio.Lock())
    monkeypatch.setattr(ResourceEngine, "_free", {})
    monkeypatch.setattr(ResourceEngine, "_acquired", {})


def test_close_ends_free_and_acquired_resources_despite_failures(engine_store):
    free = [FakeResource(), FakeResource(fail=True), FakeResource()]
    acquired = FakeResource()
    ResourceEngine._free["webshop|local"] = list(free)
    ResourceEngine._acquired["rollout-1"] = (acquired, "webshop|local")
    ResourceEngine._acquired["rollout-2"] = (None, "webshop|local")  # still being created

    async def run():
        start = asyncio.get_running_loop().time()
        await ResourceEngine.close()
        return asyncio.get_running_loop().time() - start

    elapsed = asyncio.run(run())
    assert all(r.ended for r in free + [acquired])
    assert ResourceEngine._free == {} and ResourceEngine._acquired == {}
    assert elapsed < 0.15  # concurrent, not 4 x 0.05 s


def test_trainer_close_resources_runs_engine_close_on_background_loop(monkeypatch):
    ray_trainer = pytest.importorskip("agentfly.verl.trainer.ppo.ray_trainer")
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    seen = {}

    async def fake_close():
        seen["loop"] = asyncio.get_running_loop()

    try:
        trainer = object.__new__(ray_trainer.RayPPOTrainer)
        trainer.bg_loop = loop
        monkeypatch.setattr(ResourceEngine, "close", fake_close)
        trainer.close_resources()
        assert seen["loop"] is loop

        async def failing_close():
            raise RuntimeError("boom")

        monkeypatch.setattr(ResourceEngine, "close", failing_close)
        trainer.close_resources()  # never raises
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def test_trainer_close_resources_without_background_loop_is_a_no_op():
    ray_trainer = pytest.importorskip("agentfly.verl.trainer.ppo.ray_trainer")
    trainer = object.__new__(ray_trainer.RayPPOTrainer)  # init_workers never ran
    trainer.close_resources()
