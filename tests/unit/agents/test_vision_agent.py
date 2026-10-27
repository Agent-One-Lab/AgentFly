import torch
from agentfly.agents import ReactAgent
from agentfly.agents.utils.inspection import print_trajectory
from agentfly.agents.utils.tokenizer import tokenize_trajectories
from agentfly.tools import answer_qa
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
        backend_config={"backend": "async_vllm"},
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
                        {
                            "type": "text",
                            "text": "There is an animal in the image. What is it? Also, search the information about this animal.",
                        },
                    ],
                },
            ]
        }
    ]

    result = await react_agent.run(max_turns=3, messages=messages, num_chains=10)
    print_trajectory(result[0])
    # Inspect tokenization directly; no rollout object is needed.
    inputs = tokenize_trajectories(
        react_agent, [segment.messages for trajectory in result for segment in trajectory.segments],
        tokenizer=react_agent.tokenizer,
    )
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    trajectory = result[0]
    print(f"reward: {trajectory.reward}")
    print(f"metrics: {trajectory.metrics}")
    print(f"metadata: {trajectory.metadata}")
