"""verl's ``SingleTurnAgentLoop`` executes caller-supplied ``prompt_ids`` verbatim instead of
re-rendering ``raw_prompt`` from text (which re-tokenizes earlier assistant turns and
diverges from AgentFly's training render)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
single_turn = pytest.importorskip("agentfly.verl.experimental.agent_loop.single_turn_agent_loop")
SingleTurnAgentLoop = single_turn.SingleTurnAgentLoop


class _FakeServerManager:
    def __init__(self):
        self.seen_prompt_ids = None

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, video_data=None):
        self.seen_prompt_ids = list(prompt_ids)
        return SimpleNamespace(
            token_ids=[5, 6, 7], log_probs=None, routed_experts=None, num_preempted=None, extra_fields={}
        )


def _loop(server_manager):
    loop = SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)
    loop.server_manager = server_manager
    loop.response_length = 8
    loop.prompt_length = 64

    async def no_vision(messages):
        return {}

    async def must_not_render(*args, **kwargs):
        raise AssertionError("apply_chat_template must not run when prompt_ids are supplied")

    loop.process_vision_info = no_vision
    loop.apply_chat_template = must_not_render
    return loop


def test_prompt_ids_are_used_verbatim():
    sm = _FakeServerManager()
    loop = _loop(sm)
    ids = np.empty(1, dtype=object)
    ids[0] = [1, 2, 3, 4]  # rows arrive as entries of an object array
    out = asyncio.run(loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], tools=[], prompt_ids=ids[0]))
    assert sm.seen_prompt_ids == [1, 2, 3, 4]
    assert out.prompt_ids == [1, 2, 3, 4]
    assert out.response_ids == [5, 6, 7]


def test_without_prompt_ids_the_template_path_still_runs():
    sm = _FakeServerManager()
    loop = _loop(sm)

    async def render(messages, tools=None, images=None, videos=None):
        return [9, 9]

    loop.apply_chat_template = render
    out = asyncio.run(loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], tools=[]))
    assert sm.seen_prompt_ids == [9, 9] and out.prompt_ids == [9, 9]
