from ..core import Context
from ..envs.python_env import PythonSandboxSpec
from .reward_base import reward


@reward(name="code_reward_test")
async def code_reward_test(prediction: str, context: Context) -> dict:
    """
    Run code in the rollout's Python sandbox and return reward.
    Uses the same sandbox as the code tool (context.acquire_resource with scope=rollout).
    Caller must pass context when invoking this reward.
    """
    try:
        env = await context.acquire_resource(
            spec=PythonSandboxSpec, scope="global", backend="local"
        )
        result = await env.step(prediction)
        return {"reward": 1.0, "output": result}
    except Exception as e:
        return {"reward": 0.0, "output": str(e)}
