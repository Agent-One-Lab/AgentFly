from ....rewards.scienceworld_reward import scienceworld_reward
import pytest

@pytest.mark.asyncio(loop_scope="session")
async def test_scienceworld_reward():
    prediction = "Task not completed"
    reward = await scienceworld_reward(prediction, id="test")
    assert reward["reward"] == 0.0
    await scienceworld_reward.release(id="test")
