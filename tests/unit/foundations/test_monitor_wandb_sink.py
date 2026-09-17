"""The wandb sink must make each point self-describing on its custom axis.

Events reach the sink from an async queue, so another logger (the trainer's own
``wandb.log``) can commit a wandb step between two of our calls. A metric logged
without its axis value in the same payload is then paired with whatever axis value was
last committed, silently shifting every point.
"""

import sys
import types

import pytest

from agentfly.utils.monitor import MetricEvent, WandbSink

pytestmark = pytest.mark.asyncio


class FakeRun:
    pass


class FakeWandb(types.ModuleType):
    """Minimal stand-in recording what a sink logs, in order."""

    def __init__(self):
        super().__init__("wandb")
        self.run = FakeRun()
        self.logged = []
        self.defined = []
        self.histograms = []

    def define_metric(self, name, step_metric=None):
        self.defined.append((name, step_metric))

    def log(self, payload, step=None, commit=False):
        self.logged.append((dict(payload), step, commit))

    def Histogram(self, sequence):  # noqa: N802 — mirrors wandb's class name
        self.histograms.append(list(sequence))
        return ("histogram", tuple(sequence))


@pytest.fixture
def fake_wandb(monkeypatch):
    module = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


async def test_scalar_carries_its_axis_value_in_the_same_payload(fake_wandb):
    sink = WandbSink(project="test")

    for step, value in ((4, 0.1), (5, 0.2)):
        await sink.log(MetricEvent(
            kind="scalar", name="agent/rollout/reward", value=value,
            x=step, x_name="agent/rollout/step",
        ))

    assert [payload for payload, _, _ in fake_wandb.logged] == [
        {"agent/rollout/reward": 0.1, "agent/rollout/step": 4},
        {"agent/rollout/reward": 0.2, "agent/rollout/step": 5},
    ]
    # The axis is declared once per (metric, axis) pair, not on every event.
    assert fake_wandb.defined == [
        ("agent/rollout/step", None),
        ("agent/rollout/reward", "agent/rollout/step"),
    ]


async def test_histogram_events_are_converted_and_keep_their_axis(fake_wandb):
    sink = WandbSink(project="test")

    await sink.log(MetricEvent(
        kind="hist", name="agent/rollout/group/pass_rate_hist", value=[0.0, 0.5, 1.0],
        x=3, x_name="agent/rollout/step",
    ))

    payload, step, _ = fake_wandb.logged[0]
    assert fake_wandb.histograms == [[0.0, 0.5, 1.0]]
    assert payload["agent/rollout/group/pass_rate_hist"] == ("histogram", (0.0, 0.5, 1.0))
    assert payload["agent/rollout/step"] == 3
    assert step is None


async def test_events_without_a_custom_axis_still_use_the_wandb_step(fake_wandb):
    sink = WandbSink(project="test")

    await sink.log(MetricEvent(kind="scalar", name="loss", value=1.5, step=11))

    assert fake_wandb.logged == [({"loss": 1.5}, 11, False)]
    assert fake_wandb.defined == []


async def test_nothing_is_logged_without_an_active_run(fake_wandb):
    fake_wandb.run = None
    sink = WandbSink(project="test")

    await sink.log(MetricEvent(kind="scalar", name="loss", value=1.5, x=1))

    assert fake_wandb.logged == []
