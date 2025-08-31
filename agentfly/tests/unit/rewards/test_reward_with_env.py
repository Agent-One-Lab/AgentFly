from ....rewards.code_reward import code_reward_test
import pytest

@pytest.mark.asyncio(loop_scope="session")
async def test_code_reward_test():
    code = "print('Hello, World!')"
    reward = await code_reward_test(code, id="test")
    assert reward["reward"] == 1.0
    assert reward["output"] == "Hello, World!\n"
    await code_reward_test.release("test")

