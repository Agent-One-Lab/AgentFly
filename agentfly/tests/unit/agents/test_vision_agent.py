import torch
from ....agents.react.react_agent import ReactAgent
from ....tools import answer_qa
import pytest


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_vision_agent():
    tools = [answer_qa]

    task_info = "Answer the question based on the image."

    react_agent = ReactAgent(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        tools=tools,
        template="qwen2.5-vl",
        task_info=task_info,
        backend="async_vllm"
    )

    messages = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        },
                        {"type": "text", "text": "There is an animal in the image. What is it? Also, search the information about this animal."}
                    ],
                },
            ]
        }
    ]


    await react_agent.run(
        max_turns=3,
        messages=messages,
        num_chains=10
    )
    messages_list = react_agent.get_messages()
    messages = messages_list[0]['messages']
    for message in messages:
        print(f"{message['role']}: {message['content']}")
    inputs, other_info_list = react_agent.tokenize_trajectories()
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    other_info = other_info_list[0]
    for key, value in other_info.items():
        print(f"{key}: {value}")