"""Rollout routing and the compact output of AgentFly's one-row text turns."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from agentfly.utils.llm_backends.llm_backends import AsyncVerlBackend, _render_executor, extract_response_ids
from agentfly.verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
)


class Rows:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def select_idxs(self, idxs):
        return list(idxs)


def manager(workers=8):
    m = AgentLoopManager.__new__(AgentLoopManager)
    m.agent_loop_workers = [object()] * workers
    m._next_worker = 0
    return m


def test_one_row_calls_rotate_across_workers():
    m = manager()
    used = [m._distribute_batch_round_robin(Rows(1))[1][0] for _ in range(17)]
    assert used == [i % 8 for i in range(17)]


def test_a_batch_still_covers_every_row_once():
    m = manager(3)
    m._distribute_batch_round_robin(Rows(1))
    chunks, used, original = m._distribute_batch_round_robin(Rows(7))
    assert sorted(original) == list(range(7))
    assert sorted(i for w in used for i in chunks[w]) == list(range(7))
    assert sorted(len(chunks[w]) for w in used) == [2, 2, 3]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")


def worker(tokenizer, processor=None):
    w = AgentLoopWorker.__new__(AgentLoopWorker)
    w.tokenizer = tokenizer
    w.processor = processor
    w.rollout_config = SimpleNamespace(prompt_length=64, response_length=32)
    w.reward_loop_worker_handles = None
    w.distillation_enabled = False
    w.reward_score = None

    async def nothing(*args, **kwargs):
        return None

    w._compute_score = nothing
    w._compute_teacher_logprobs = nothing
    return w


def loop_output(multi_modal_data=None):
    return AgentLoopOutput(
        prompt_ids=[11, 12, 13, 14], response_ids=[21, 22, 23], response_mask=[1, 1, 1], response_logprobs=[-0.1, -0.2, -0.3],
        multi_modal_data=multi_modal_data or {}, num_turns=2, metrics=AgentLoopMetrics(), extra_fields={},
    )


def test_compact_row_reads_back_the_same_response(tokenizer):
    w = worker(tokenizer)
    kwargs = {"raw_prompt": [], "prompt_ids": [11, 12, 13, 14], "_single_row": True}
    compact = asyncio.run(w._agent_loop_postprocess(loop_output(), False, **kwargs))
    full = asyncio.run(w._agent_loop_postprocess(loop_output(), False, **{**kwargs, "_single_row": False}))
    assert compact.input_ids.shape[1] == 7 and full.input_ids.shape[1] == 96
    read = lambda o: extract_response_ids({"responses": o.response_ids, "attention_mask": o.attention_mask, "prompts": o.prompt_ids})
    assert read(compact) == read(full) == [[21, 22, 23]]
    assert compact.prompt_ids[compact.attention_mask[:, :4].bool()].tolist() == [11, 12, 13, 14]
    assert compact.multi_modal_inputs is None
    assert compact.response_logprobs.shape[1] == 3


@pytest.mark.parametrize(
    "change",
    [
        {"_single_row": False},
        {"prompt_ids": None},
    ],
)
def test_other_rows_keep_the_full_path(tokenizer, change):
    w = worker(tokenizer)
    kwargs = {"prompt_ids": [1], "_single_row": True, **change}
    assert not w._compact_output(loop_output(), kwargs)


@pytest.mark.parametrize("data", [{"images": ["image"]}, {"videos": ["video"]}])
def test_vision_rows_keep_the_full_path(tokenizer, data):
    w = worker(tokenizer)
    assert not w._compact_output(SimpleNamespace(multi_modal_data=data), {"prompt_ids": [1], "_single_row": True})


def test_reward_or_teacher_workers_keep_the_full_path(tokenizer):
    w = worker(tokenizer)
    w.reward_loop_worker_handles = [object()]
    assert not w._compact_output(loop_output(), {"prompt_ids": [1], "_single_row": True})
    w = worker(tokenizer)
    w.distillation_enabled = True
    assert not w._compact_output(loop_output(), {"prompt_ids": [1], "_single_row": True})


def test_render_off_the_loop_gives_identical_ids(tokenizer):
    backend = AsyncVerlBackend.__new__(AsyncVerlBackend)
    backend.tokenizer, backend.template, backend.model_name = tokenizer, "action-agent-qwen35", "Qwen/Qwen3.5-4B"
    rows = [
        [{"role": "system", "content": "s " * (i + 1)}, {"role": "user", "content": f"question {i} " * 300}]
        for i in range(32)
    ]
    expected = [backend._render_prompt_ids([r], None)[0] for r in rows]

    async def render():
        loop = asyncio.get_running_loop()
        return await asyncio.gather(*(loop.run_in_executor(_render_executor(), backend._render_prompt_ids, [r], None) for r in rows))

    assert [x[0] for x in asyncio.run(render())] == expected
    assert threading.current_thread().name == "MainThread"
