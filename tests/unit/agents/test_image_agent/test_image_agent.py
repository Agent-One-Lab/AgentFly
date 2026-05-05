from agentfly.agents import ImageEditingAgent
import pytest


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_image_agent():
    agent = ImageEditingAgent(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        template="qwen2.5-vl-system-tool",
        backend_config={"backend": "async_vllm"},
        streaming="console",
    )
    # --8<-- [start:messages_list]
    messages_list = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        },
                        {
                            "type": "text",
                            "text": "Find what animal is in the image, then inpaint it with a cat.",
                        },
                    ],
                }
            ]
        }
    ]
    # --8<-- [end:messages_list]
    # --8<-- [start:agent_run_print]
    await agent.run(
        messages=messages_list,
        max_turns=4,
        num_chains=1,
        enable_streaming=True,
    )
    agent.print_messages(index=0)
    # --8<-- [end:agent_run_print]

    inputs, _ = agent.tokenize_trajectories()
    for k, v in inputs.items():
        print(f"{k}: {v.shape}")
