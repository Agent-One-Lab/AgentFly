import asyncio

import pytest

from agentfly.agents.rollout.chain import ChainRollout
from agentfly.agents.rollout.events import ChainEnded


@pytest.mark.asyncio
async def test_run_async_respects_max_concurrent_chains(monkeypatch):
    rollout = ChainRollout()

    # Minimal attributes needed by ChainRollout.run_async
    rollout.tools = []

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def _fake_run_chain(
        self,
        chain_id,
        first_node,
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
        first_node.is_terminal = True
        # _run_chains reads final state from here, not from the events.
        self.chains[chain_id] = chain
        self.current_nodes[chain_id] = first_node
        async with lock:
            in_flight -= 1
        yield ChainEnded(chain_id=chain_id)

    monkeypatch.setattr(
        rollout, "_run_chain", _fake_run_chain.__get__(rollout, ChainRollout)
    )

    await rollout.run_async(
        messages=[{"role": "user", "content": "hi"}],
        max_turns=1,
        num_chains=5,
        max_concurrent_chains=2,
    )

    assert max_in_flight <= 2

