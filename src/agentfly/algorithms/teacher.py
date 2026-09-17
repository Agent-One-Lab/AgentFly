"""Client for a teacher-scoring service: per-token log-probabilities of existing token sequences.

On-policy distillation needs a teacher model's log-probability of every token the student
sampled. A teacher deployment (a vLLM server) publishes an *endpoints file*; this client reads it
and asks the server to score token-id sequences with ``/v1/completions`` (``max_tokens=1``,
``prompt_logprobs=0``): a prefill-only pass returning ``log p(token_i | tokens_<i)``.

The module depends only on numpy and aiohttp (no verl, no torch), so offline analyses can use it
as well as the trainer::

    client = TeacherClient("~/Research/deploy/teachers/qwen38_27b.json",
                           expect_model="Qwen/Qwen3.8-27B-FP8")
    logprobs, stats = client.score_blocking([[151644, 872, 198], ...])
    # logprobs[k][i] = log p(seq_k[i] | seq_k[:i]); position 0 is NaN

Endpoints file (written atomically by the deployment)::

    {"name": "qwen38_27b", "model": "Qwen/Qwen3.8-27B-FP8", "revision": "017b9c7a...",
     "max_model_len": 96000, "tp_size": 2, "dp_size": 4, "urls": ["http://10.0.4.94:31000"],
     "status": "ready", "slurm_job": "1211864", "written_at": "2026-09-16T18:55:26Z"}

``status`` is ``starting`` while the service (re)starts — including after a Slurm requeue, which
moves it to a new node and IP — ``ready`` once it serves, and ``stopped`` after shutdown.

Failure policy: a call either returns every requested score or raises. Connection errors,
timeouts, 5xx responses and a not-ready endpoints file are waited out (with backoff, re-reading
the file so a moved service is found again) for at most ``max_wait_s`` of continuous outage;
other 4xx responses and malformed responses raise immediately.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

STATUSES = ("starting", "ready", "stopped")


class TeacherError(RuntimeError):
    """The teacher service returned something unusable (not retried)."""


class TeacherUnavailableError(TeacherError):
    """The teacher could not be reached, or was not ready, for longer than ``max_wait_s``."""


class TeacherIdentityError(TeacherError):
    """The deployment serves a different model or revision than the client expects."""


class SequenceTooLongError(TeacherError, ValueError):
    """A sequence does not fit the teacher's context (checked before any request)."""


@dataclass(frozen=True)
class TeacherEndpoints:
    """A parsed and validated endpoints file."""

    name: str
    model: str
    max_model_len: int
    status: str
    urls: Tuple[str, ...]
    revision: Optional[str] = None
    tp_size: int = 1
    dp_size: int = 1
    slurm_job: Optional[str] = None
    written_at: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @classmethod
    def load(cls, path: "str | Path") -> "TeacherEndpoints":
        path = Path(path).expanduser()
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read teacher endpoints file {path}: {exc}") from exc
        missing = [k for k in ("name", "model", "max_model_len", "status", "urls") if k not in raw]
        if missing:
            raise ValueError(f"Teacher endpoints file {path} is missing {missing}")
        if raw["status"] not in STATUSES:
            raise ValueError(f"Teacher endpoints file {path}: status {raw['status']!r} not in {STATUSES}")
        urls = tuple(str(u).rstrip("/") for u in raw["urls"])
        if raw["status"] == "ready" and not urls:
            raise ValueError(f"Teacher endpoints file {path} is ready but lists no urls")
        return cls(
            name=str(raw["name"]), model=str(raw["model"]), max_model_len=int(raw["max_model_len"]),
            status=raw["status"], urls=urls, revision=raw.get("revision"),
            tp_size=int(raw.get("tp_size", 1)), dp_size=int(raw.get("dp_size", 1)),
            slurm_job=None if raw.get("slurm_job") is None else str(raw["slurm_job"]),
            written_at=raw.get("written_at"),
        )


@dataclass
class ScoreStats:
    """What one :meth:`TeacherClient.score` call did."""

    sequences: int = 0
    unique: int = 0
    tokens: int = 0
    requests: int = 0
    retries: int = 0
    seconds: float = 0.0


class _Retryable(Exception):
    """Internal: a transient failure worth waiting out."""


def _first_leaf(error: BaseException) -> BaseException:
    """The first non-group exception inside (possibly nested) exception groups."""
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error


class TeacherClient:
    """Scores token sequences against the teacher named by an endpoints file.

    Args:
        endpoints_file: path to the deployment's endpoints file.
        expect_model / expect_revision: when set, the endpoints file must match them.
        concurrency: concurrent requests per URL (default ``2 x dp_size``).
        request_timeout_s: per-request timeout.
        max_wait_s: longest continuous outage (unreachable, 5xx, or not ready) to wait out.
        poll_interval_s: how often to re-read a not-ready endpoints file.
        backoff_initial_s / backoff_max_s: retry backoff bounds.
    """

    def __init__(
        self,
        endpoints_file: "str | Path",
        *,
        expect_model: Optional[str] = None,
        expect_revision: Optional[str] = None,
        concurrency: Optional[int] = None,
        request_timeout_s: float = 3600.0,
        max_wait_s: float = 1800.0,
        poll_interval_s: float = 15.0,
        backoff_initial_s: float = 5.0,
        backoff_max_s: float = 60.0,
    ):
        self.endpoints_file = Path(endpoints_file).expanduser()
        self.expect_model = expect_model
        self.expect_revision = expect_revision
        self.concurrency = concurrency
        self.request_timeout_s = request_timeout_s
        self.max_wait_s = max_wait_s
        self.poll_interval_s = poll_interval_s
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s
        self._verified_urls: set = set()

    # ---- public API ------------------------------------------------------------------------

    async def score(
        self, sequences: Sequence[Sequence[int]], topk: int = 0
    ) -> Tuple[List[np.ndarray], ScoreStats]:
        """Per-token log-probabilities for each sequence, in input order.

        Returns ``(logprobs, stats)`` where ``logprobs[k]`` is a float32 array of
        ``len(sequences[k])`` with position 0 set to NaN. Identical sequences are scored once.
        """
        if topk:
            raise NotImplementedError("top-k teacher log-probabilities are not supported yet")
        import aiohttp  # lazy: optional for importers that never score

        started = time.monotonic()
        keys = [tuple(int(t) for t in seq) for seq in sequences]
        if any(len(k) == 0 for k in keys):
            raise ValueError("cannot score an empty sequence")
        unique: Dict[Tuple[int, ...], int] = {}
        for key in keys:
            unique.setdefault(key, len(unique))
        stats = ScoreStats(sequences=len(keys), unique=len(unique), tokens=sum(len(k) for k in unique))

        endpoints = await self._wait_ready(time.monotonic())
        longest = max((len(k) for k in unique), default=0)
        if longest > endpoints.max_model_len - 1:
            raise SequenceTooLongError(
                f"sequence of {longest} tokens does not fit teacher {endpoints.name!r} "
                f"(max_model_len {endpoints.max_model_len}, one position is needed for the dummy output)"
            )

        results: List[Optional[np.ndarray]] = [None] * len(unique)
        state = {"endpoints": endpoints}
        semaphores: Dict[str, asyncio.Semaphore] = {}
        refresh_lock = asyncio.Lock()

        def semaphore(url: str, ep: TeacherEndpoints) -> asyncio.Semaphore:
            limit = self.concurrency or max(1, 2 * ep.dp_size)
            return semaphores.setdefault(url, asyncio.Semaphore(limit))

        async def refresh(outage_started: float) -> None:
            async with refresh_lock:
                state["endpoints"] = await self._wait_ready(outage_started)

        timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:

            async def work(key: Tuple[int, ...], slot: int) -> None:
                backoff = self.backoff_initial_s
                outage_started: Optional[float] = None
                while True:
                    ep = state["endpoints"]
                    url = ep.urls[slot % len(ep.urls)]
                    try:
                        await self._verify(session, url, ep)
                        async with semaphore(url, ep):
                            stats.requests += 1
                            results[unique[key]] = await self._request(session, url, ep.name, key)
                        return
                    except _Retryable as exc:
                        now = time.monotonic()
                        outage_started = outage_started if outage_started is not None else now
                        if now - outage_started > self.max_wait_s:
                            raise TeacherUnavailableError(
                                f"teacher {ep.name!r} unavailable for {now - outage_started:.0f}s: {exc}"
                            ) from exc
                        stats.retries += 1
                        logger.warning("teacher request to %s failed (%s); retrying in %.1fs", url, exc, backoff)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, self.backoff_max_s)
                        await refresh(outage_started)

            try:
                async with asyncio.TaskGroup() as group:
                    for slot, key in enumerate(unique):
                        group.create_task(work(key, slot))
            except BaseExceptionGroup as group_error:
                # The first failure cancels the rest; surface it as itself, not as a group.
                raise _first_leaf(group_error) from None

        stats.seconds = time.monotonic() - started
        return [results[unique[k]] for k in keys], stats

    def score_blocking(
        self, sequences: Sequence[Sequence[int]], topk: int = 0
    ) -> Tuple[List[np.ndarray], ScoreStats]:
        """Synchronous :meth:`score` on a private event loop (e.g. from a trainer worker thread)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.score(sequences, topk=topk))
        raise RuntimeError("score_blocking() called from a running event loop; await score() instead")

    # ---- internals -------------------------------------------------------------------------

    def _load_checked(self) -> TeacherEndpoints:
        endpoints = TeacherEndpoints.load(self.endpoints_file)
        if self.expect_model is not None and endpoints.model != self.expect_model:
            raise TeacherIdentityError(
                f"teacher {endpoints.name!r} serves {endpoints.model!r}, expected {self.expect_model!r}"
            )
        if self.expect_revision is not None and endpoints.revision != self.expect_revision:
            raise TeacherIdentityError(
                f"teacher {endpoints.name!r} is at revision {endpoints.revision!r}, "
                f"expected {self.expect_revision!r}"
            )
        return endpoints

    async def _wait_ready(self, outage_started: float) -> TeacherEndpoints:
        while True:
            endpoints = self._load_checked()
            if endpoints.ready:
                return endpoints
            waited = time.monotonic() - outage_started
            if waited > self.max_wait_s:
                raise TeacherUnavailableError(
                    f"teacher {endpoints.name!r} not ready after {waited:.0f}s (status {endpoints.status!r})"
                )
            await asyncio.sleep(self.poll_interval_s)

    async def _verify(self, session, url: str, endpoints: TeacherEndpoints) -> None:
        """Check once per URL that the server serves the endpoints file's model name."""
        if url in self._verified_urls:
            return
        import aiohttp

        try:
            async with session.get(f"{url}/v1/models") as response:
                if response.status >= 500:
                    raise _Retryable(f"/v1/models HTTP {response.status}")
                response.raise_for_status()
                served = [m.get("id") for m in (await response.json()).get("data", [])]
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
            raise _Retryable(f"{type(exc).__name__}: {exc}") from exc
        if endpoints.name not in served:
            raise TeacherIdentityError(f"{url} serves {served}, expected {endpoints.name!r}")
        self._verified_urls.add(url)

    async def _request(self, session, url: str, name: str, ids: Tuple[int, ...]) -> np.ndarray:
        import aiohttp

        body = {"model": name, "prompt": list(ids), "max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0}
        try:
            async with session.post(f"{url}/v1/completions", json=body) as response:
                if response.status >= 500:
                    raise _Retryable(f"HTTP {response.status}: {(await response.text())[:200]}")
                if response.status != 200:
                    raise TeacherError(f"{url} HTTP {response.status}: {(await response.text())[:300]}")
                payload = await response.json()
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
            raise _Retryable(f"{type(exc).__name__}: {exc}") from exc

        try:
            entries = payload["choices"][0]["prompt_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TeacherError(f"{url} response has no prompt_logprobs") from exc
        if len(entries) != len(ids):
            raise TeacherError(f"{url} returned {len(entries)} prompt_logprobs for {len(ids)} tokens")
        logprobs = np.full(len(ids), np.nan, dtype=np.float32)
        for i, entry in enumerate(entries):
            if entry is None:
                continue
            hit = entry.get(str(ids[i]))
            if hit is None:
                raise TeacherError(f"{url} returned no log-probability for token {ids[i]} at position {i}")
            logprobs[i] = hit["logprob"] if isinstance(hit, dict) else hit
        return logprobs
