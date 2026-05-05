import pytest
from agentfly.agents import ReactAgent
from agentfly.tools import async_dense_retrieve_api
from agentfly.tools import answer_qa


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_react_agent_parse_run():
    tools = [async_dense_retrieve_api, answer_qa]
    agent = ReactAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend_config={"backend": "async_vllm"},
    )

    responses = [
        """Thought: I need to search for information.
Action: google_search
Input: {"query": "test query"}"""
    ]

    result = agent.parse(responses, tools)
    print(result)
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert (
        "Thought: I need to search for information." in result[0]["content"][0]["text"]
    )
    assert len(result[0]["tool_calls"]) == 1
    assert result[0]["tool_calls"][0]["function"]["name"] == "google_search"
    assert result[0]["tool_calls"][0]["function"]["arguments"] == {
        "query": "test query"
    }

    messages = [
        {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
    ]
    run_result = await agent.run(max_turns=4, messages=messages, num_chains=1)
    print(run_result.trajectories)
