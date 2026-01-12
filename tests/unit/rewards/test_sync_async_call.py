import pytest
from agentfly.rewards import reward


@pytest.mark.asyncio(loop_scope="session")
async def test_sync_async_call():
    @reward(name="test_reward")
    async def test_reward(prediction: str):
        return 1.0
    
    result = await test_reward(prediction="Hello, World!")
    assert result['reward'] == 1.0

    @reward(name="test_reward_sync")
    def test_reward_sync(prediction: str):
        return 1.0
    
    result = test_reward_sync(prediction="Hello, World!")
    assert result['reward'] == 1.0