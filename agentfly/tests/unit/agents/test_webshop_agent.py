import pytest
from ....agents.react.react_agent import ReactAgent
from ....tools.src.webshop.tools import webshop_browser
from ....tools.src.react.tools import answer
from ....rewards import webshop_reward


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_webshop_agent_call():
    tools = [webshop_browser, answer]
    agent = ReactAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        reward_fn=webshop_reward,
        template="qwen2.5",
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
            "asin": "B07FYPSNH8",
            "category": "grocery",
            "query": "beverages",
            "name": "OWYN - 100% Vegan Plant-Based Protein Shakes | Cold Brew Coffee, 12 Fl Oz | Dairy-Free, Gluten-Free, Soy-Free, Tree Nut-Free, Egg-Free, Allergy-Free, Vegetarian",
            "product_category": "Grocery & Gourmet Food \u203a Beverages \u203a Bottled Beverages, Water & Drink Mixes \u203a Meal Replacement & Protein Drinks \u203a Protein Drinks",
            "instruction_text": "i am looking for a gluten free, 100% vegan plant based protein shake that is soy-free, and price lower than 40.00 dollars",
            "attributes": [
            "gluten free"
            ],
            "price_upper": 40.0,
            "goal_options": [],
            "weight": 1,
            "task_id": 0,
        },
    ]


    await agent.run(
            max_turns=8,
            messages=messages,
            num_chains=4
        )

    messages = agent.get_messages()
    print(messages)
