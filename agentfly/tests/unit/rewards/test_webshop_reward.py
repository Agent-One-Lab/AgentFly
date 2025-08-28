from ....rewards.webshop_reward import webshop_reward
import pytest

@pytest.mark.asyncio(loop_scope="session")
async def test_webshop_reward():
    prediction = "Thank you for shopping with us"
    reward = await webshop_reward(prediction, task_id=0, id="test_webshop_reward")
    assert reward["reward"] == 0.0
    await webshop_reward.release(id="test_webshop_reward")
