from agentfly.agents import HFAgent
from agentfly.tools import calculate, answer_math
import pytest
from agentfly.rewards import math_reward_string_equal

@pytest.mark.asyncio
async def test_quick_example():
    # messages = [
    #     {
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": "What is the result of 1 + 1?"
    #             }
    #         ]
    #     }
    # ]
    # messages = [{"role": "user", "content": "What is the result of 1 + 1?"}]
    messages = {
        "messages": [
            {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"}
        ],
        "answer": "72"
    }
    agent = HFAgent(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        tools=[calculate],
        reward_fn=math_reward_string_equal,
        template="qwen2.5",
        backend="async_vllm",
    )
    await agent.run(
        messages=messages,
        max_turns=3,
        num_chains=5
    )

    trajectories = agent.trajectories
    print(trajectories)
    print(agent.rewards)