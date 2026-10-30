"""``run(use_token_ids=...)`` switches carrying the sampled token ids on and off.

With the default (True) each generated assistant message carries the backend's sampled
``token_ids``, which tokenization (chat-bricks) splices instead of re-encoding the text.
With False the rollout leaves them off, so every later render re-encodes from text.
"""
import inspect
from types import SimpleNamespace

import pytest

from agentfly.agents.agent_base import BaseAgent
from agentfly.agents.rollout.base import Rollout
from agentfly.agents.rollout.loop.generation import generate_response
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.rollout.strategies.step_rollout import StepRollout
from agentfly.agents.types import RunResult

SAMPLED = [27, 26865, 29, 64, 151645]


class _FakeAgent:
    max_model_len = None
    default_tool_call_source = "parser"

    def __init__(self, response):
        self.response = response

    def skills_payload(self):
        return None

    async def generate_async(self, messages, **kwargs):
        return self.response

    def parse(self, response_texts, tool_calls=None, context=None, **kwargs):
        return [{"role": "assistant", "content": response_texts[0]}]


def _generate(response, **kwargs):
    step = SimpleNamespace(messages=SimpleNamespace(messages=[{"role": "user", "content": "hi"}]))
    return generate_response(
        _FakeAgent(response), None, step, None, 0, "c0", {}, SimpleNamespace(metadata={}), **kwargs
    )


@pytest.mark.asyncio
async def test_sampled_ids_are_carried_by_default():
    msg, _ = await _generate({"response_texts": ["<action>x</action>"], "response_ids": [SAMPLED]})
    assert msg["token_ids"] == SAMPLED


@pytest.mark.asyncio
async def test_sampled_ids_are_left_off_when_disabled():
    msg, _ = await _generate(
        {"response_texts": ["<action>x</action>"], "response_ids": [SAMPLED]}, use_token_ids=False
    )
    assert "token_ids" not in msg


@pytest.mark.asyncio
async def test_ids_returned_on_the_raw_message_are_dropped_when_disabled():
    response = {
        "response_texts": ["<action>x</action>"],
        "response_dict": {"choices": [{"message": {"content": "<action>x</action>", "token_ids": SAMPLED}}]},
    }
    msg, _ = await _generate(response, use_token_ids=False)
    assert "token_ids" not in msg


@pytest.mark.asyncio
@pytest.mark.parametrize("use_token_ids", [True, False])
async def test_agent_run_forwards_use_token_ids(use_token_ids):
    seen = {}

    class Capture(Rollout):
        async def run(self, agent, messages, max_turns, use_token_ids=True, **kwargs):
            seen["use_token_ids"] = use_token_ids
            return RunResult(trajectories=[])

    agent = SimpleNamespace(
        _preprocess_messages=lambda messages: messages,
        _preprocess_backends=lambda: None,
        _postprocess_backends=lambda: None,
        postprocess_trajectories=lambda trajectories: trajectories,
    )
    await BaseAgent.run(agent, messages=[], max_turns=1, rollout=Capture(), use_token_ids=use_token_ids)
    assert seen["use_token_ids"] is use_token_ids


@pytest.mark.parametrize("strategy", [Rollout, StepRollout, ChainRollout])
def test_strategies_accept_use_token_ids_defaulting_to_true(strategy):
    assert inspect.signature(strategy.run).parameters["use_token_ids"].default is True
    assert strategy.use_token_ids is True
