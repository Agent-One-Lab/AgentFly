import sys
sys.path.insert(0, '/research/d1/gds/xmhu23/AgentFly')

from agents.agents.specialized.image_agent.image_agent import ImageEditingAgent
import pytest

@pytest.mark.asyncio
async def test_image_agent():
    agent = ImageEditingAgent(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        template="qwen2.5-vl-system-tool",
        backend="async_vllm",
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
    await agent.run_async(
        start_messages=messages_list,
        max_steps=4,
        num_chains=1,
        enable_streaming=True
    )
    agent.print_messages(index=0)

    inputs, _ = agent.tokenize_trajectories()
    for k, v in inputs.items():
        print(f"{k}: {v.shape}")
