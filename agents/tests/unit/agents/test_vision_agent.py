from agents.agents.react.react_agent import ReactAgent
from agents.tools import google_search_serper, answer

import pytest

@pytest.mark.asyncio(loop_scope="session")
async def test_vision_agent():
    tools = [google_search_serper, answer]

    task_info = "Use web search to get answers."

    react_agent = ReactAgent(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        tools=tools,
        template="qwen2.5-vl",
        task_info=task_info,
        backend="async_vllm",
        debug=True
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


    await react_agent.run_async(
        max_steps=3,
        start_messages=messages,
        num_chains=10
    )

    inputs = react_agent.tokenize_trajectories(return_action_mask=True)
    print(inputs)