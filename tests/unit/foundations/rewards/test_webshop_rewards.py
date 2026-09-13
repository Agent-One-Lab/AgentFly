"""WebShop rewards: ``webshop_episode_reward`` is verl-agent's success reward (10 / 0);
``webshop_reward`` keeps the dense score. No container needed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentfly.rewards.webshop_reward import webshop_episode_reward, webshop_reward


def _ctx(score=None, error=None):
    class Env:
        async def step(self, action, task_id=None):
            assert action == "get_reward"
            if error:
                raise error
            return {"observation": "Thank you for shopping with us!", "reward": score}

    async def acquire(**kwargs):
        return Env()

    return SimpleNamespace(acquire_resource=acquire, metadata={})


@pytest.mark.asyncio
@pytest.mark.parametrize("score,reward,won", [(1.0, 10.0, 1.0), (0.75, 0.0, 0.0), (0.0, 0.0, 0.0)])
async def test_episode_reward_is_verl_agent_success_scale(score, reward, won):
    out = await webshop_episode_reward(context=_ctx(score), task_id=3)
    assert out == {"reward": reward, "trajectory/accuracy": won, "task_score": score}


@pytest.mark.asyncio
async def test_episode_reward_env_error_counts_as_failure():
    out = await webshop_episode_reward(context=_ctx(error=RuntimeError("no purchase")), task_id=3)
    assert out == {"reward": 0.0, "trajectory/accuracy": 0.0, "task_score": 0.0}


@pytest.mark.asyncio
async def test_dense_reward_unchanged():
    out = await webshop_reward(final_response="", context=_ctx(0.75), task_id=3)
    assert out["reward"] == 0.75
