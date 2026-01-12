import os
from agentfly.agents import ImageEditingAgent
from agentfly.agents.llm_backends import ClientConfig, AsyncVLLMConfig
import pytest

@pytest.mark.asyncio
async def test_image_agent_client():
    agent = ImageEditingAgent(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        backend="client",
        backend_config=ClientConfig(
            base_url="http://localhost:8000/v1",
        ),
        streaming="console"
    )
    messages_list = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                        },
                        {
                            "type": "text",
                            "text": "Find what animal is in the image, then inpaint it with a cat."
                        }
                    ]
                }
            ]
        }
    ]
    await agent.run(
        messages=messages_list,
        max_turns=4,
        num_chains=1,
        enable_streaming=True
    )
    agent.print_messages(index=0)


@pytest.mark.asyncio
async def test_image_agent_openai():
    agent = ImageEditingAgent(
        model_name_or_path="gpt-5-mini",
        backend="client",
        backend_config=ClientConfig(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        streaming="console"
    )
    messages_list = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                        },
                        {
                            "type": "text",
                            "text": "Find what animal is in the image, then inpaint it with a cat."
                        }
                    ]
                }
            ]
        }
    ]
    await agent.run(
        messages=messages_list,
        max_turns=5,
        num_chains=1,
        enable_streaming=True
    )
    agent.print_messages(index=0)


@pytest.mark.asyncio
async def test_image_agent_async_vllm():
    agent = ImageEditingAgent(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        template="qwen2.5-vl-system-tool",
        backend="async_vllm",
        backend_config=AsyncVLLMConfig(
            pipeline_parallel_size=4, # Use pp = 4
            gpu_memory_utilization=0.5

        ),
        streaming="console"
    )
    messages_list = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            # "image_url": {
                            #     "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                            # }
                            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                        },
                        {
                            "type": "text",
                            "text": "Find what animal is in the image, then inpaint it with a cat."
                        }
                    ]
                }
            ]
        }
    ]
    await agent.run(
        messages=messages_list,
        max_turns=4,
        num_chains=1,
        enable_streaming=True
    )
    agent.print_messages(index=0)