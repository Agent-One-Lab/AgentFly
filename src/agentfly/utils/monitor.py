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
            with open(default_file, "w"):
                pass

            self.log_files = {
                "default": open(default_file, "a", buffering=1, encoding="utf-8")
            }
        else:
            self.log_files = {}

        self._lock = asyncio.Lock()

    async def log(self, evt: MetricEvent) -> None:
        evt_name = evt.name.replace("/", "-")
        if evt_name not in self.log_files:
            file_name = os.path.join(self.directory, f"{evt_name}.jsonl")
            with open(file_name, "w"):
                pass
            self.log_files[evt_name] = open(
                file_name, "a", buffering=1, encoding="utf-8"
            )
        file_obj = self.log_files[evt_name]

        async with self._lock:
            file_obj.write(
                json.dumps(serialize_for_json(asdict(evt)), ensure_ascii=False) + "\n"
            )

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
                    wandb.log(
                        {
                            evt.name: wandb.plot.line(
                                table, evt.x_name, evt.name, title=evt.name
                            )
                        },
                        commit=evt.commit,
                    )
                elif evt.kind == "text":
                    if evt.name not in self.tables:
                        self.tables[evt.name] = wandb.Table(
                            columns=["step", "text"], log_mode="INCREMENTAL"
                        )
                    self.tables[evt.name].add_data(evt.x, evt.value)
                    wandb.log({evt.name: self.tables[evt.name]}, commit=evt.commit)
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


class Monitor:
    """Singleton helper controlling the consumer task and registered sinks."""

    _sinks: Dict[str, BaseSink] = {}
    _queue: Optional["asyncio.Queue[MetricEvent | None]"] = None
    _queue_loop: Optional[asyncio.AbstractEventLoop] = None
    _consumer_task: Optional[asyncio.Task[None]] = None
    _running: bool = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def _ensure_queue(cls) -> "asyncio.Queue[MetricEvent | None]":
        """Ensure queue exists and is bound to the current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop - create queue without binding (will bind on first use)
            if cls._queue is None:
                cls._queue = asyncio.Queue()
                cls._queue_loop = None
            return cls._queue

        # Check if queue needs to be recreated for current event loop
        if cls._queue is None or cls._queue_loop is not current_loop:
            # Recreate queue for current event loop
            # Note: We can't migrate events from old queue, but that's acceptable
            # since events are best-effort and the old loop is closed anyway
            cls._queue = asyncio.Queue()
            cls._queue_loop = current_loop

        return cls._queue

    @classmethod
    def ensure_started(cls) -> None:
        # Check if consumer task is still running
        if cls._running and cls._consumer_task is not None:
            try:
                # Check if task is done or cancelled
                if cls._consumer_task.done():
                    # Task completed/cancelled, need to restart
                    cls._running = False
                    cls._consumer_task = None
                else:
                    # Task is still running, nothing to do
                    return
            except (RuntimeError, AttributeError, Exception):
                # Task might be from a different event loop that was closed
                # or the task object might be invalid
                cls._running = False
                cls._consumer_task = None

        # Ensure queue is bound to current event loop
        cls._ensure_queue()

        # Create new consumer task
        try:
            loop = asyncio.get_running_loop()
            cls._consumer_task = loop.create_task(
                cls._consumer_loop(), name="monitor-consumer"
            )
            cls._running = True
        except RuntimeError:
            # No running event loop - this shouldn't happen in normal usage
            # but we'll handle it gracefully
            print(
                "[Monitor] Warning: No running event loop found. Monitor consumer not started."
            )
            cls._running = False
            cls._consumer_task = None

    @classmethod
    async def shutdown(cls) -> None:
        """Flush sinks and stop background task (call at program exit)."""

        if not cls._running:
            return
        # Ensure queue exists
        queue = cls._ensure_queue()
        # send sentinel
        await queue.put(None)
        await cls._consumer_task
        for sink in list(cls._sinks.values()):
            with contextlib.suppress(Exception):
                await sink.close()
        cls._sinks.clear()
        cls._running = False
        cls._queue = None
        cls._queue_loop = None

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

    @classmethod
    async def _consumer_loop(cls) -> None:
        queue = cls._ensure_queue()
        while True:
            evt = await queue.get()
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
        while not queue.empty():
            queue.get_nowait()


def emit(evt: MetricEvent) -> None:
    """Enqueue an event for asynchronous processing (non‑blocking)."""

    Monitor.ensure_started()
    queue = Monitor._ensure_queue()
    try:
        queue.put_nowait(evt)
    except (asyncio.QueueFull, RuntimeError) as e:
        # QueueFull: extremely unlikely – drop oldest
        # RuntimeError: queue not bound to current event loop (shouldn't happen in normal usage)
        if isinstance(e, RuntimeError) and "bound to a different event loop" in str(e):
            # Queue is bound to different loop, recreate it
            Monitor._queue = None
            Monitor._queue_loop = None
            queue = Monitor._ensure_queue()
            try:
                queue.put_nowait(evt)
            except asyncio.QueueFull:
                queue.get_nowait()
                queue.put_nowait(evt)
        elif isinstance(e, asyncio.QueueFull):
            queue.get_nowait()
            queue.put_nowait(evt)
        else:
            # Re-raise other RuntimeErrors
            raise
