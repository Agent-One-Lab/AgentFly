import asyncio
from types import SimpleNamespace

import pytest

from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.rollout.events import ChainEnded


@pytest.mark.asyncio
async def test_run_respects_max_concurrent_chains(monkeypatch):
    """``ChainRollout`` fans out chains but caps concurrency at ``max_concurrent_chains``.

    Post-refactor, the rollout is a standalone strategy object that reaches agent
    state through ``self.agent`` (the host). The concurrency loop (``_run_chains``)
    only needs ``host.tools`` before it hands each chain to ``_run_chain`` (which we
    monkeypatch to a fast no-op), so a minimal fake host suffices.
    """
    rollout = ChainRollout()
    rollout.agent = SimpleNamespace(tools=[], prompt_tools=lambda: [])  # host surface used by _run_chains

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def _fake_run_chain(
        self,
        chain_id,
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
        # _run_chains reads final state from here, not from the events.
        self.chains[chain_id] = chain
        self.current_steps[chain_id] = first_step
        async with lock:
            in_flight -= 1
        yield ChainEnded(chain_id=chain_id)

    monkeypatch.setattr(
        rollout, "_run_chain", _fake_run_chain.__get__(rollout, ChainRollout)
    )

    async for _ev in rollout._run_chains(
        messages=[{"role": "user", "content": "hi"}],
        max_turns=1,
        num_chains=5,
        max_concurrent_chains=2,
    ):
        pass

    assert max_in_flight <= 2
