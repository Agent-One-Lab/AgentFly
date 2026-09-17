"""Teacher-scoring client against a local fake of vLLM's prompt_logprobs API."""

import asyncio
import contextlib
import json
import socket
import threading

import numpy as np
import pytest
from aiohttp import web

from agentfly.algorithms.teacher import (
    SequenceTooLongError,
    TeacherClient,
    TeacherEndpoints,
    TeacherError,
    TeacherIdentityError,
    TeacherUnavailableError,
)


def logprob_of(token):
    return -token / 100.0


class FakeTeacher:
    """Minimal vLLM /v1/completions + /v1/models; records requests and in-flight concurrency."""

    def __init__(self, name="teacher", *, status=200, fail_first=0, delay=0.0,
                 drop_token=False, short=False):
        self.name, self.status, self.fail_first, self.delay = name, status, fail_first, delay
        self.drop_token, self.short = drop_token, short
        self.prompts, self.in_flight, self.max_in_flight = [], 0, 0
        self.runner, self.url = None, None

    async def completions(self, request):
        body = await request.json()
        self.prompts.append(body["prompt"])
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            if self.fail_first > 0:
                self.fail_first -= 1
                return web.Response(status=503, text="warming up")
            if self.status != 200:
                return web.Response(status=self.status, text="bad request")
            ids = body["prompt"]
            entries = [None] + [{str(t): {"logprob": logprob_of(t)}} for t in ids[1:]]
            if self.drop_token:
                entries[-1] = {"999999": {"logprob": -1.0}}
            if self.short:
                entries = entries[:-1]
            return web.json_response({"choices": [{"prompt_logprobs": entries, "text": ""}]})
        finally:
            self.in_flight -= 1

    async def models(self, request):
        return web.json_response({"data": [{"id": self.name}]})

    async def start(self):
        app = web.Application()
        app.router.add_post("/v1/completions", self.completions)
        app.router.add_get("/v1/models", self.models)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    async def stop(self):
        await self.runner.cleanup()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_endpoints(path, urls, *, name="teacher", model="org/Teacher", status="ready",
                    max_model_len=1000, revision="abc", dp_size=1):
    path.write_text(json.dumps({
        "name": name, "model": model, "revision": revision, "max_model_len": max_model_len,
        "tp_size": 1, "dp_size": dp_size, "urls": urls, "status": status,
        "slurm_job": "1", "written_at": "now",
    }))


@contextlib.asynccontextmanager
async def teachers(*fakes):
    started = [await f.start() for f in fakes]
    try:
        yield started
    finally:
        for f in started:
            await f.stop()


def fast_client(path, **kw):
    kw.setdefault("poll_interval_s", 0.02)
    kw.setdefault("backoff_initial_s", 0.02)
    kw.setdefault("backoff_max_s", 0.05)
    return TeacherClient(path, **kw)


@pytest.mark.asyncio
async def test_returns_logprob_of_each_token_with_position_zero_nan(tmp_path):
    async with teachers(FakeTeacher()) as (t,):
        write_endpoints(tmp_path / "e.json", [t.url])
        (lp,), stats = await fast_client(tmp_path / "e.json").score([[5, 6, 7]])
    assert np.isnan(lp[0])
    np.testing.assert_allclose(lp[1:], [logprob_of(6), logprob_of(7)])
    assert lp.dtype == np.float32
    assert (stats.sequences, stats.unique, stats.tokens, stats.requests, stats.retries) == (1, 1, 3, 1, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["short", "drop_token"])
async def test_malformed_responses_raise_without_retry(tmp_path, fault):
    async with teachers(FakeTeacher(**{fault: True})) as (t,):
        write_endpoints(tmp_path / "e.json", [t.url])
        with pytest.raises(TeacherError):
            await fast_client(tmp_path / "e.json").score([[1, 2, 3]])
        assert len(t.prompts) == 1


@pytest.mark.asyncio
async def test_identical_sequences_are_scored_once_and_results_shared(tmp_path):
    async with teachers(FakeTeacher()) as (t,):
        write_endpoints(tmp_path / "e.json", [t.url])
        lps, stats = await fast_client(tmp_path / "e.json").score([[1, 2], [3, 4], [1, 2]])
    assert sorted(map(tuple, t.prompts)) == [(1, 2), (3, 4)]
    np.testing.assert_array_equal(lps[0], lps[2])
    assert (stats.sequences, stats.unique, stats.requests) == (3, 2, 2)


@pytest.mark.asyncio
async def test_round_robin_across_urls_respects_per_url_concurrency(tmp_path):
    async with teachers(FakeTeacher(delay=0.05), FakeTeacher(delay=0.05)) as (a, b):
        write_endpoints(tmp_path / "e.json", [a.url, b.url])
        seqs = [[i, i + 1] for i in range(1, 13, 2)]
        await fast_client(tmp_path / "e.json", concurrency=2).score(seqs)
    assert len(a.prompts) == len(b.prompts) == 3
    assert a.max_in_flight <= 2 and b.max_in_flight <= 2


@pytest.mark.asyncio
async def test_default_concurrency_is_twice_dp_size(tmp_path):
    async with teachers(FakeTeacher(delay=0.05)) as (t,):
        write_endpoints(tmp_path / "e.json", [t.url], dp_size=2)
        await fast_client(tmp_path / "e.json").score([[i, i + 1] for i in range(10)])
    assert t.max_in_flight == 4


@pytest.mark.asyncio
async def test_waits_while_the_service_is_starting(tmp_path):
    path = tmp_path / "e.json"
    async with teachers(FakeTeacher()) as (t,):
        write_endpoints(path, [], status="starting")

        async def become_ready():
            await asyncio.sleep(0.1)
            write_endpoints(path, [t.url])

        waiter = asyncio.create_task(become_ready())
        (lp,), _ = await fast_client(path, max_wait_s=5).score([[1, 2]])
        await waiter
    assert lp[1] == pytest.approx(logprob_of(2))


@pytest.mark.asyncio
async def test_follows_the_service_to_a_new_address(tmp_path):
    path = tmp_path / "e.json"
    async with teachers(FakeTeacher()) as (t,):
        write_endpoints(path, [f"http://127.0.0.1:{free_port()}"])  # requeued: old address is dead

        async def moved():
            await asyncio.sleep(0.1)
            write_endpoints(path, [t.url])

        mover = asyncio.create_task(moved())
        (lp,), stats = await fast_client(path, max_wait_s=5).score([[1, 2]])
        await mover
    assert stats.retries >= 1
    assert lp[1] == pytest.approx(logprob_of(2))


@pytest.mark.asyncio
async def test_server_errors_are_retried(tmp_path):
    async with teachers(FakeTeacher(fail_first=2)) as (t,):
        write_endpoints(tmp_path / "e.json", [t.url])
        _, stats = await fast_client(tmp_path / "e.json", max_wait_s=5).score([[1, 2]])
    assert stats.retries == 2 and stats.requests == 3


@pytest.mark.asyncio
async def test_gives_up_after_max_wait(tmp_path):
    write_endpoints(tmp_path / "e.json", [f"http://127.0.0.1:{free_port()}"])
    with pytest.raises(TeacherUnavailableError):
        await fast_client(tmp_path / "e.json", max_wait_s=0.2).score([[1, 2]])


@pytest.mark.asyncio
async def test_not_ready_for_too_long_raises(tmp_path):
    write_endpoints(tmp_path / "e.json", [], status="starting")
    with pytest.raises(TeacherUnavailableError, match="not ready"):
        await fast_client(tmp_path / "e.json", max_wait_s=0.1).score([[1, 2]])


@pytest.mark.asyncio
async def test_client_errors_are_not_retried(tmp_path):
    async with teachers(FakeTeacher(status=400)) as (t,):
        write_endpoints(tmp_path / "e.json", [t.url])
        with pytest.raises(TeacherError, match="HTTP 400"):
            await fast_client(tmp_path / "e.json").score([[1, 2]])
    assert len(t.prompts) == 1


@pytest.mark.asyncio
async def test_identity_is_checked_against_file_and_server(tmp_path):
    async with teachers(FakeTeacher(name="other")) as (t,):
        write_endpoints(tmp_path / "e.json", [t.url])
        with pytest.raises(TeacherIdentityError, match="expected 'org/Wrong'"):
            await fast_client(tmp_path / "e.json", expect_model="org/Wrong").score([[1, 2]])
        with pytest.raises(TeacherIdentityError, match="expected revision|expected 'zzz'"):
            await fast_client(tmp_path / "e.json", expect_revision="zzz").score([[1, 2]])
        with pytest.raises(TeacherIdentityError, match="serves \\['other'\\]"):
            await fast_client(tmp_path / "e.json").score([[1, 2]])
        assert t.prompts == []


@pytest.mark.asyncio
async def test_over_length_sequences_raise_before_any_request(tmp_path):
    async with teachers(FakeTeacher()) as (t,):
        write_endpoints(tmp_path / "e.json", [t.url], max_model_len=4)
        with pytest.raises(SequenceTooLongError):
            await fast_client(tmp_path / "e.json").score([[1, 2], [1, 2, 3, 4]])
    assert t.prompts == []


@pytest.mark.asyncio
async def test_topk_is_reserved(tmp_path):
    write_endpoints(tmp_path / "e.json", ["http://127.0.0.1:1"])
    with pytest.raises(NotImplementedError):
        await fast_client(tmp_path / "e.json").score([[1, 2]], topk=5)


def test_score_blocking_works_from_a_worker_thread(tmp_path):
    fake = FakeTeacher()
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def serve():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(fake.start())
        ready.set()
        loop.run_forever()

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    assert ready.wait(10)
    try:
        write_endpoints(tmp_path / "e.json", [fake.url])
        out = {}
        worker = threading.Thread(
            target=lambda: out.update(result=fast_client(tmp_path / "e.json").score_blocking([[3, 4]]))
        )
        worker.start()
        worker.join(30)
        (lp,), stats = out["result"]
        assert lp[1] == pytest.approx(logprob_of(4)) and stats.requests == 1
    finally:
        asyncio.run_coroutine_threadsafe(fake.stop(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        server.join(10)


def test_endpoints_file_validation(tmp_path):
    path = tmp_path / "e.json"
    path.write_text(json.dumps({"name": "t", "model": "m", "status": "ready", "urls": ["http://x"]}))
    with pytest.raises(ValueError, match="missing"):
        TeacherEndpoints.load(path)
    write_endpoints(path, [], status="ready")
    with pytest.raises(ValueError, match="lists no urls"):
        TeacherEndpoints.load(path)
    write_endpoints(path, ["http://x/"], status="unknown")
    with pytest.raises(ValueError, match="status"):
        TeacherEndpoints.load(path)
    write_endpoints(path, ["http://x/"])
    assert TeacherEndpoints.load(path).urls == ("http://x",)
