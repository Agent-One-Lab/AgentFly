import asyncio
from types import SimpleNamespace

import pytest

from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.rollout.events import EpisodeEnded


@pytest.mark.asyncio
async def test_run_respects_max_concurrent_chains(monkeypatch):
    """``ChainRollout`` fans out chains but caps concurrency at ``max_concurrent_episodes``.

    The rollout reaches the agent through the ``agent`` argument. The concurrency loop
    (``stream``) only needs ``agent.prompt_tools()`` before it hands each chain to
    ``run_episode`` (which we monkeypatch to a fast no-op), so a minimal fake agent suffices.
    """
    rollout = ChainRollout()
    agent = SimpleNamespace(tools=[], prompt_tools=lambda: [])  # agent surface used by stream

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def fake_run_episode(
        self,
        agent,
        episode_id,
        first_step,
        chain,
        tools,
        max_turns,
        generation_config,
        context_config=None,
    ):
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        first_step.is_terminal = True
        # stream reads final state from here, not from the events.
        self.chains[episode_id] = chain
        self.current_steps[episode_id] = first_step
        async with lock:
            in_flight -= 1
        yield EpisodeEnded(episode_id=episode_id)

    monkeypatch.setattr(
        rollout, "run_episode", fake_run_episode.__get__(rollout, ChainRollout)
    )

    async for _ev in rollout.stream(
        agent,
        messages=[{"role": "user", "content": "hi"}],
        max_turns=1,
        group_size=5,
        max_concurrent_episodes=2,
    ):
        pass

    assert max_in_flight <= 2
