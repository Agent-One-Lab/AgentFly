import pytest

@pytest.mark.asyncio
async def test_quick_example():

    from agentfly.agents import HFAgent
    from agentfly.tools import calculator
    agent = HFAgent(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        tools=[calculator],
        template="qwen2.5",
        backend="async_vllm",
    )
    messages = [{"role": "user", "content": "What is the result of 1 + 1?"}]
    await agent.run(
        messages=messages,
        max_turns=3,
        num_chains=1
    )

    trajectories = agent.trajectories
    print(trajectories)
    print(agent.rewards)


@pytest.mark.asyncio
async def define_tool_reward():
    from agentfly.tools import tool
    from sympy import simplify, sympify, Rational

    @tool(name="calculator", description="Calculate the result of a mathematical expression.")
    def calculator(expression: str):
        try:
            expr = sympify(expression)
            result = simplify(expr)

            # Check if the result is a number
            if result.is_number:
                # If the result is a rational number, return as a fraction
                if isinstance(result, Rational):
                    return str(result)
                # If the result is a floating point number, format to remove redundant zeros
                else:
                    return "{:g}".format(float(result))
            else:
                return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    print(calculator.schema)
    
    result = calculator(expression="1 + 1")
    print(result)

    from agentfly.agents import HFAgent
    from agentfly.tools import calculator
    from agentfly.rewards import math_reward_string_equal

    agent = HFAgent(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        tools=[calculator],
        template="qwen2.5",
        reward_fn=math_reward_string_equal,
        backend="async_vllm",
    )

    messages = {
        "messages": [
            {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"}
        ],
        "answer": "72"
    }
    await agent.run(
        messages=messages,
        max_turns=3,
        num_chains=5 # Generate 5 trajectories for the query
    )

    trajectories = agent.trajectories
    rewards = agent.rewards
    print(trajectories)
    print(rewards)
