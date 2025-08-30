import pytest
from ....agents.react.react_agent import ReactAgent, parse_react_step
from ....tools.src.search.google_search import google_search_serper
from ....tools import answer_qa

def test_parse_react_step():
    # Test with a valid ReAct step
    text = """Thought: I need to find information about Python.
Action: google_search
Input: {"query": "Python programming language"}"""
    
    result = parse_react_step(text)
    assert result["thought"] == "I need to find information about Python."
    assert result["action"] == "google_search"
    assert result["input"] == '{"query": "Python programming language"}'
    
    # Test with missing components
    text_missing = "Thought: I'm thinking about something."
    result_missing = parse_react_step(text_missing)
    assert result_missing["thought"] == "I'm thinking about something."
    assert result_missing["action"] is None
    assert result_missing["input"] is None


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_react_agent_parse_run():
    tools = [google_search_serper, answer_qa]
    agent = ReactAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend="async_vllm"
    )
    
    responses = ["""Thought: I need to search for information.
Action: google_search
Input: {"query": "test query"}"""]
    
    result = agent.parse(responses, tools)
    print(result) 
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "Thought: I need to search for information." in result[0]["content"][0]["text"]
    assert len(result[0]["tool_calls"]) == 1
    assert result[0]["tool_calls"][0]["function"]["name"] == "google_search"
    assert result[0]["tool_calls"][0]["function"]["arguments"] == {"query": "test query"}

    messages = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ]
        }
    ]
    await agent.run(
        max_turns=4,
        messages=messages,
        num_chains=1
    )
    messages_list = agent.get_messages()
    print(messages_list[0])
    