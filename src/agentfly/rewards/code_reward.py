from ..envs.python_env import PythonSandboxEnv
from .reward_base import reward


@reward(name="code_reward_test", env_cls=PythonSandboxEnv, pool_size=16)
async def code_reward_test(prediction: str, env: PythonSandboxEnv) -> dict:
    try:
        result = await env.step(prediction)
        return {"reward": 1.0, "output": result}
    except Exception as e:
        return {"reward": 0.0, "output": str(e)}
