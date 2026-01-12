from agentfly.agents import ReactAgent, CodeAgent
from agentfly.tools import answer_qa, code_interpreter, asyncdense_retrieve, webshop_browser, scienceworld_explorer
from agentfly.rewards import math_reward, scienceworld_reward, webshop_reward
import pytest


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_code_agent():
    agent = CodeAgent(
        model_name_or_path="Agent-One/Qwen2.5-3B-Code-Code",
        tools=[code_interpreter],
        reward_fn=math_reward,
        template="qwen2.5-no-system-tool",
        backend="async_vllm",
        streaming="console"
    )
    await agent.run(
        messages=[
            {
                "messages": [
                    {"role": "user", "content": "If $f(x) = \frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction."}
                ],
                "question": "If $f(x) = \frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction.",
                "answer": "14/3"
            }
        ],
        max_turns=4,
        num_chains=1,
        enable_streaming=True
    )
    print(f"Trajectories: {agent.trajectories}")
    print(f"Rewards: {agent.rewards}")


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_react_vqa_agent():
    agent = ReactAgent(
        model_name_or_path="Agent-One/Qwen2.5-VL-3B-VQA-React",
        tools=[answer_qa],
        template="qwen2.5-vl",
        backend="async_vllm",
        streaming="console",
    )
    await agent.run(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in the image?"},
                    {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
                ]
            }
        ],
        max_turns=4,
        num_chains=1,
        enable_streaming=True
    )
    print(agent.trajectories)

async def test_react_vqa_retrieval_agent():
    agent = ReactAgent(
        model_name_or_path="Agent-One/Qwen2.5-VL-3B-Retrieval-React",
        tools=[asyncdense_retrieve, answer_qa],
        template="qwen2.5-vl",
        backend="async_vllm",
        streaming="console",
    )
    await agent.run(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Give me the information about the animal in the image."},
                    {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
                ]
            }
        ],
        max_turns=4,
        num_chains=1,
        enable_streaming=True
    )
    print(agent.trajectories)


async def test_react_scienceworld_agent():
    agent = ReactAgent(
        model_name_or_path="Agent-One/Qwen2.5-7B-ScienceWorld-React",
        tools=[scienceworld_explorer],
        reward_fn=scienceworld_reward,
        template="qwen2.5-no-system-tool",
        backend="async_vllm",
        streaming="console",
    )
    await agent.run(
        messages={
            "messages": [
                {
                    "role": "user",
                    "content": "Your task is to determine whether pointed leaf shape is a dominant or recessive trait in the unknown C plant. If the trait is dominant, focus on the blue box. If the trait is recessive, focus on the orange box."
                }
            ],
            "question": "Your task is to determine whether pointed leaf shape is a dominant or recessive trait in the unknown C plant. If the trait is dominant, focus on the blue box. If the trait is recessive, focus on the orange box.",
            "task_name": "mendelian-genetics-unknown-plant",
            "variation_idx": 281
        },
        max_turns=12,
        num_chains=1,
        enable_streaming=True
    )
    print(f"Trajectories: {agent.trajectories}")
    print(f"Rewards: {agent.rewards}")



async def test_react_webshop_agent():
    agent = ReactAgent(
        model_name_or_path="Agent-One/Qwen2.5-7B-Webshop-React",
        tools=[webshop_browser],
        reward_fn=webshop_reward,
        template="qwen2.5-no-system-tool",
        backend="async_vllm",
        streaming="console",
    )
    await agent.run(
        messages={
            "messages": [
                {
                    "role": "user",
                    "content": "i want a super soft jay franco disney minnie mouse twin bed set, and price lower than 120.00 dollars"
                }
            ],
            "asin": "B07RT28DLG",
            "category": "garden",
            "query": "furniture sets",
            "name": "Jay Franco Disney Minnie Mouse Lashes Bed Set, Full",
            "product_category": "Home & Kitchen \u203a Bedding \u203a Kids' Bedding \u203a Comforters & Sets \u203a Comforter Sets",
            "instruction_text": "i want a super soft jay franco disney minnie mouse twin bed set, and price lower than 120.00 dollars",
            "attributes": [
                "super soft"
            ],
            "price_upper": 120.0,
            "goal_options": [
                "twin"
            ],
            "weight": 1,
            "task_id": 56,
            "question": "i want a super soft jay franco disney minnie mouse twin bed set, and price lower than 120.00 dollars"
        },
        max_turns=12,
        num_chains=1,
        enable_streaming=True
    )
    print(f"Trajectories: {agent.trajectories}")
    print(f"Rewards: {agent.rewards}")

