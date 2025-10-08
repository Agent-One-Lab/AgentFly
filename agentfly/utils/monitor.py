"""Light-weight, decoupled monitoring layer for AgentRL.

Usage example (see __main__ at bottom):
    >>> from monitoring import emit, MetricEvent, Monitor, JsonlSink, WandbSink
    >>> Monitor.add_sink("jsonl", JsonlSink("run.jsonl"))
    >>> Monitor.add_sink("wandb", WandbSink(project="agentrl"))
    >>> emit(MetricEvent("scalar", "reward/episode", 1.0, step=0))
    >>> await Monitor.shutdown()

Importing *only* `emit` + `MetricEvent` in your modules avoids wandb/file I/O
coupling and lets you toggle sinks at runtime.
"""

import abc
import asyncio
from collections import defaultdict
import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
import wandb
from PIL import Image
import io
import base64
import numpy as np

@dataclass(slots=True)
class MetricEvent:
    """A single observation produced by any module.

    Attributes
    ----------
    kind        Category of metric (scalar | hist | text | resource).
    name        Fully‑qualified metric name (e.g. "reward/qa_f1").
    value       Numeric / text payload.
    step        Integer training step or episode counter.
    timestamp   Unix seconds (auto‑filled if omitted).
    tags        Arbitrary key/value pairs for filtering (e.g. run_id, module).
    sinks       List of sink names to send this event to. If None, sends to all sinks.
    """

    kind: Literal["scalar", "hist", "text", "resource", "list"]
    name: str
    value: Any
    sinks: Optional[List[str]] = None
    step: Optional[int] = None
    x: Optional[int] = None
    x_name: Optional[str] = "x_axis"
    commit: bool = False
    timestamp: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.time()


class BaseSink(abc.ABC):
    """Abstract writer backend."""

    @abc.abstractmethod
    async def log(self, evt: MetricEvent) -> None:  # pragma: no cover
        ...

    async def flush(self) -> None:  # optional override
        pass

    async def close(self) -> None:  # optional override
        await self.flush()

    # handy for printing readable name in errors
    def __repr__(self) -> str:  # noqa: D401
        return f"<{self.__class__.__name__}>"


def serialize_for_json(obj):
    if isinstance(obj, np.ndarray):
        # Convert numpy array to list
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        # Convert numpy scalars to Python types
        return obj.item()
    elif isinstance(obj, Image.Image):
        # Convert image to base64 string
        buffer = io.BytesIO()
        obj.save(buffer, format="PNG")
        return {"__image__": base64.b64encode(buffer.getvalue()).decode("utf-8")}
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_for_json(i) for i in obj)
    elif isinstance(obj, bytes):
        return {"__image__": base64.b64encode(obj).decode("utf-8")}
    else:
        return obj  # leave other types as-is

class JsonlSink(BaseSink):
    """Append events as JSON-Lines - human & machine friendly."""

    def __init__(self, directory: str) -> None:
        os.makedirs(os.path.dirname(directory) or ".", exist_ok=True)
        self.directory = directory
        if os.path.isdir(directory):
            default_file = os.path.join(directory, "default.jsonl")
            with open(default_file, 'w') as f:
                pass

            self.log_files = {"default": open(default_file, "a", buffering=1, encoding="utf-8")}
        else:
            self.log_files = {}

        self._lock = asyncio.Lock()

    async def log(self, evt: MetricEvent) -> None:
        evt_name = evt.name.replace("/", "-")
        if evt_name not in self.log_files:
            file_name = os.path.join(self.directory, f"{evt_name}.jsonl")
            with open(file_name, 'w') as f:
                pass
            self.log_files[evt_name] = open(file_name, "a", buffering=1, encoding="utf-8")
        file_obj = self.log_files[evt_name]

        async with self._lock:
            file_obj.write(json.dumps(serialize_for_json(asdict(evt)), ensure_ascii=False) + "\n")

    async def flush(self) -> None:
        for file_obj in self.log_files.values():
            file_obj.flush()

    async def close(self) -> None:
        await super().close()
        for file_obj in self.log_files.values():
            file_obj.close()


class WandbSink(BaseSink):
    """Weights & Biases backend (lazy import)."""

    def __init__(self, project: str, **wandb_init_kwargs: Any) -> None:  # noqa: D401
        import importlib

        # self.wandb = importlib.import_module("wandb")  # lazy, keeps wandb optional
        # if wandb.run is None:
        #     wandb.init(project=project, **wandb_init_kwargs)
        self._defined_axes: Set[Tuple[str, str]] = set()
        self.tables: Dict[str, wandb.Table] = {}

    async def log(self, evt: MetricEvent) -> None:  # pragma: no cover
        """
        Log the event to wandb.
        """
        if wandb.run is not None:
            payload = {evt.name: evt.value, **evt.tags}
            if evt.x is not None:
                if evt.kind == "list":
                    data = [[x, y] for x, y in zip(evt.x, evt.value)]
                    table = wandb.Table(data=data, columns=[evt.x_name, evt.name])
                    wandb.log({
                        evt.name: wandb.plot.line(table, evt.x_name, evt.name, title=evt.name)
                    }, commit=evt.commit)
                elif evt.kind == "text":
                    if evt.name not in self.tables:
                        self.tables[evt.name] = wandb.Table(columns=["step", "text"], log_mode="INCREMENTAL")
                    self.tables[evt.name].add_data(evt.x, evt.value)
                    wandb.log({
                        evt.name: self.tables[evt.name]
                    }, commit=evt.commit)
                else:
                    key = (evt.name, evt.x_name)
                    if key not in self._defined_axes:
                        wandb.define_metric(evt.x_name)
                        wandb.define_metric(evt.name, step_metric=evt.x_name)
                        self._defined_axes.add(key)
                    wandb.log(payload, commit=evt.commit)
            else:
                wandb.log(payload, step=evt.step, commit=evt.commit)

    async def flush(self) -> None:  # pragma: no cover
        wandb.log({}, commit=True)  # forces step commit
        wandb.flush()

    async def close(self) -> None:  # pragma: no cover
        await super().close()
        wandb.finish()


# Example of a wrapper sink that filters kinds/names without touching producers
class FilterSink(BaseSink):
    """Wrap another sink and allow include/exclude rules."""

    def __init__(
        self,
        wrapped: BaseSink,
        include_kinds: Optional[List[str]] = None,
        exclude_kinds: Optional[List[str]] = None,
    ) -> None:
        self.wrapped = wrapped
        self.include = set(include_kinds or [])
        self.exclude = set(exclude_kinds or [])

    async def log(self, evt: MetricEvent) -> None:
        if self.include and evt.kind not in self.include:
            return
        if evt.kind in self.exclude:
            return
        await self.wrapped.log(evt)

    async def flush(self) -> None:
        await self.wrapped.flush()

    async def close(self) -> None:
        await self.wrapped.close()



class Monitor:
    """Singleton helper controlling the consumer task and registered sinks."""

    _sinks: Dict[str, BaseSink] = {}
    _queue: "asyncio.Queue[MetricEvent | None]" = asyncio.Queue()
    _consumer_task: Optional[asyncio.Task[None]] = None
    _running: bool = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def ensure_started(cls) -> None:
        if cls._running:
            return
        cls._consumer_task = asyncio.create_task(cls._consumer_loop(), name="monitor-consumer")
        cls._running = True

    @classmethod
    async def shutdown(cls) -> None:
        """Flush sinks and stop background task (call at program exit)."""

        if not cls._running:
            return
        # send sentinel
        await cls._queue.put(None)
        await cls._consumer_task
        for sink in list(cls._sinks.values()):
            with contextlib.suppress(Exception):
                await sink.close()
        cls._sinks.clear()
        cls._running = False

    # ── sink management ─────────────────────────────────────────────────────

    @classmethod
    def add_sink(cls, name: str, sink: BaseSink) -> None:
        cls._sinks[name] = sink

    @classmethod
    def remove_sink(cls, name: str) -> None:
        sink = cls._sinks.pop(name, None)
        if sink is None:
            return
        # enqueue coroutine to close the sink without blocking caller
        async def _close() -> None:
            await sink.close()
        asyncio.create_task(_close())

    # ── core consumer ───────────────────────────────────────────────────────

    @classmethod
    async def _consumer_loop(cls) -> None:
        while True:
            evt = await cls._queue.get()
            if evt is None:  # sentinel
                break
            for sink_name, sink in list(cls._sinks.items()):
                # Check if this sink should receive this event
                if evt.sinks is not None and sink_name not in evt.sinks:
                    continue
                try:
                    await sink.log(evt)
                except Exception as exc:
                    print(f"[Monitor] Sink {sink!r} failed: {exc}")
        # drain any remaining events (best‑effort)
        while not cls._queue.empty():
            cls._queue.get_nowait()



def emit(evt: MetricEvent) -> None:
    """Enqueue an event for asynchronous processing (non‑blocking)."""

    Monitor.ensure_started()
    try:
        Monitor._queue.put_nowait(evt)
    except asyncio.QueueFull:  # extremely unlikely – drop oldest
        Monitor._queue.get_nowait()
        Monitor._queue.put_nowait(evt)


class ResourcePoller:
    """Emit process RSS/CPU every *interval* seconds."""

    def __init__(self, interval: float = 10.0, *, run_id: str | None = None):
        import psutil  # heavyweight; keep local to avoid hard dep for everybody

        self.psutil = psutil
        self.interval = interval
        self.run_id = run_id
        self._task: Optional[asyncio.Task[None]] = None

    # API -------------------------------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="resource-poller")

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    # internals --------------------------------------------------------------
    async def _loop(self) -> None:
        proc = self.psutil.Process(os.getpid())
        while True:
            rss_mb = proc.memory_info().rss / 1e6
            cpu = proc.cpu_percent(interval=None)  # % since last call
            emit(
                MetricEvent(
                    kind="resource",
                    name="memory/rss_mb",
                    value=rss_mb,
                    tags={"run": self.run_id} if self.run_id else {},
                )
            )
            emit(
                MetricEvent(
                    kind="resource",
                    name="cpu/percent",
                    value=cpu,
                    tags={"run": self.run_id} if self.run_id else {},
                )
            )
            await asyncio.sleep(self.interval)



async def _demo() -> None:  # pragma: no cover
    """Run with `python -m monitoring` to see events flowing."""

    # 1. sinks ----------------------------------------------------------------
    Monitor.add_sink("jsonl", JsonlSink("demo_metrics.jsonl"))

    try:
        Monitor.add_sink("wandb", WandbSink(project="agentrl-demo"))
    except ModuleNotFoundError:
        print("wandb not installed - skipping WandbSink")

    # 2. start poller ---------------------------------------------------------
    poller = ResourcePoller(interval=5.0)
    poller.start()

    # 3. produce some fake scalar metrics ------------------------------------
    for step in range(20):
        reward = 1.0 - (step / 20)
        emit(MetricEvent("scalar", "reward/step", reward, step=step))
        await asyncio.sleep(0.5)

    # 4. graceful shutdown ----------------------------------------------------
    poller.stop()
    await Monitor.shutdown()


if __name__ == "__main__":
    asyncio.run(_demo())
