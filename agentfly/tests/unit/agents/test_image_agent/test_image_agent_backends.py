from .....agents import ImageEditingAgent
from .....agents.llm_backends import ClientConfig
import pytest

@pytest.mark.asyncio
async def test_image_agent():
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

    inputs, _ = agent.tokenize_trajectories()
    for k, v in inputs.items():
        print(f"{k}: {v.shape}")