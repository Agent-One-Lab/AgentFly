from agentfly.envs import WarmPool
from agentfly.envs import PythonSandboxEnv
import pytest

@pytest.mark.asyncio
async def test_warm_pool():
    pool = WarmPool(factory=PythonSandboxEnv.acquire, size=10)
    await pool.start()
    await pool.aclose()

