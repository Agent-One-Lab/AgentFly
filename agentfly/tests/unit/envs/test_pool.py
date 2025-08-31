from ....envs.manager.warm_pool import WarmPool
from ....envs.python_env import PythonSandboxEnv
import pytest

@pytest.mark.asyncio
async def test_warm_pool():
    pool = WarmPool(factory=PythonSandboxEnv.acquire, size=10)
    await pool.start()
    await pool.aclose()

