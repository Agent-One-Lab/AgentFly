import pytest
from ....agents.specialized.code_agent import CodeAgent
from ....tools import code_interpreter


@pytest.mark.asyncio
async def test_code_agent_end_to_end():
    tools = [code_interpreter]

    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend="async_vllm",
    )

    question1 = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."
    answer1 = "204"
    question2 = "$P(x)$ is a polynomial of degree $3n$ such that\n\\begin{eqnarray*} P(0) = P(3) = \\cdots &=& P(3n) = 2, \\\\ P(1) = P(4) = \\cdots &=& P(3n-2) = 1, \\\\ P(2) = P(5) = \\cdots &=& P(3n-1) = 0, \\quad\\text{ and }\\\\ && P(3n+1) = 730.\\end{eqnarray*}\nDetermine $n$."
    answer2 = "3"
    messages = [
        {
            "messages": [
                {"role": "user", "content": f"{question1}"}
            ],
            "question": f"{question1}",
            "answer": f"{answer1}"
        },
        {
            "messages": [
                {"role": "user", "content": f"{question2}"}
            ],
            "question": f"{question2}",
            "answer": f"{answer2}"
        }
    ]

    await agent.run(
        max_turns=4,
        messages=messages,
        num_chains=2
    )

    messages = agent.get_messages()
    print(messages)