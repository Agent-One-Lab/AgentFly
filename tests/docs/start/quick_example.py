import pytest


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_quick_example():
    # --8<-- [start:agent_init]
    from agentfly.agents import HFAgent
    from agentfly.tools import calculator

    agent = HFAgent(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        tools=[calculator],
        template="qwen2.5",
        backend_config={"backend": "async_vllm"},
    )
    # --8<-- [end:agent_init]

    # --8<-- [start:agent_run]
    messages = [{"role": "user", "content": "What is the result of 1 + 1?"}]
    result = await agent.run(
        messages=messages,
        max_turns=3,
        num_chains=1,
    )
    # --8<-- [end:agent_run]

    # --8<-- [start:agent_trajectories]
    trajectories = result.trajectories
    print(trajectories)
    # --8<-- [end:agent_trajectories]
    print(result.rewards)


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_tool_reward():
    # --8<-- [start:tool_def]
    from agentfly.tools import tool
    from sympy import simplify, sympify, Rational

    @tool(
        name="calculator",
        description="Calculate the result of a mathematical expression.",
    )
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
    # --8<-- [end:tool_def]

    # --8<-- [start:reward_def]
    from agentfly.rewards import reward
    from typing import List, Dict
    import re

    @reward(name="math_reward_string_equal")
    def math_reward_string_equal(
        prediction: str, answer: str, trajectory: List[Dict]
    ) -> float:
        def extract_last_number(s: str):
            matches = re.findall(r"\d+", s)  # find all sequences of digits
            return matches[-1] if matches else None

        tool_count = 0
        for msg in trajectory:
            if msg["role"] == "tool":
                tool_count += 1

        if tool_count < 1:
            return 0.0
        else:
            prediction = extract_last_number(prediction)

            if prediction == answer:
                return 1.0
            else:
                return 0.1
    # --8<-- [end:reward_def]

    # --8<-- [start:agent_with_reward]
    from agentfly.agents import HFAgent

    agent = HFAgent(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        tools=[calculator],
        template="qwen2.5",
        reward_fn=math_reward_string_equal,
        backend_config={"backend": "async_vllm"},
    )
    # --8<-- [end:agent_with_reward]

    # --8<-- [start:run_with_answer]
    messages = {
        "messages": [
            {
                "role": "user",
                "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            }
        ],
        "answer": "72",
    }
    result = await agent.run(
        messages=messages,
        max_turns=3,
        num_chains=5,  # Generate 5 trajectories for the query
    )
    # --8<-- [end:run_with_answer]

    # --8<-- [start:agent_results]
    trajectories = result.trajectories
    rewards = result.rewards
    print(trajectories)
    print(rewards)
    # --8<-- [end:agent_results]
