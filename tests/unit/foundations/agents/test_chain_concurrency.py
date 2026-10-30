import asyncio
from types import SimpleNamespace

import pytest

from agentfly.agents.rollout.events import EpisodeEnded
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.rollout.strategies.step_rollout import StepRollout


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [ChainRollout, StepRollout])
async def test_stream_respects_max_concurrent_episodes(monkeypatch, strategy_cls):
    """The shared ``stream`` fans out episodes but caps concurrency at
    ``max_concurrent_episodes``.

    ``run_episode`` is monkeypatched to a fast no-op, so a minimal fake agent suffices.
    """
    rollout = strategy_cls()
    agent = SimpleNamespace(tools=[], prompt_tools=lambda: [])

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def fake_run_episode(self, agent, spec, *, max_turns, generation_config, context_config=None):
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        yield EpisodeEnded(episode_id=spec.episode_id)

    monkeypatch.setattr(rollout, "run_episode", fake_run_episode.__get__(rollout, strategy_cls))

    ended = [
        event
        async for event in rollout.stream(
            agent,
            messages=[{"role": "user", "content": "hi"}],
            max_turns=1,
            group_size=5,
            max_concurrent_episodes=2,
        )
    ]

    assert len(ended) == 5
    assert max_in_flight <= 2
