"""Run an agent on the number-guessing environment: ``python examples/custom_env/run.py``."""

# --8<-- [start:run]
import asyncio

from my_task.tools import guess, guess_reward

from agentfly.agents import HFAgent
from agentfly.resources import ResourceEngine

PROMPT = (
    "I'm thinking of a whole number between 1 and 100. Find it with the guess tool, "
    "which tells you whether the number is higher or lower than your guess."
)


async def main():
    agent = HFAgent(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        tools=[guess],
        reward_fn=guess_reward,
        template="qwen2.5",
        backend_config={"backend": "async_vllm", "gpu_memory_utilization": 0.5},
    )
    tasks = [{"messages": [{"role": "user", "content": PROMPT}], "target": t} for t in (37, 81)]
    result = await agent.run(messages=tasks, max_turns=10, group_size=2)
    for traj in result:
        print(f"target={traj.metadata['target']} reward={traj.reward} guesses={traj.metrics['guesses']}")
    await ResourceEngine.close()  # end the pooled environments


if __name__ == "__main__":
    asyncio.run(main())
# --8<-- [end:run]
