import pytest
from agents.agents.react.react_agent import ReactAgent
from agents.tools.src.webshop.tools import webshop_browser
from agents.tools.src.react.tools import answer
from rewards.webshop_reward import WebshopReward


@pytest.mark.asyncio
async def test_webshop_agent_call():
    tools = [webshop_browser, answer]
    agent = ReactAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        reward_fn=WebshopReward,
        template="qwen-7b-chat",
        backend="async_vllm",
        debug=True
    )
    
    question = "i am looking for a gluten free, 100% vegan plant based protein shake that is soy-free, and price lower than 40.00 dollars"
    messages = [
        {
            "messages": [
                {"role": "user", "content": f"{question}"}
            ],
            "question": f"{question}",
        },
    ]


    await agent.run_async(
            max_steps=8,
            start_messages=messages,
            num_chains=4
        )

    messages = agent.get_messages()
    print(messages)