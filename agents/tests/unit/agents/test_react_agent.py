import pytest
from agents.agents.react.react_agent import ReactAgent, parse_react_step
from agents.tools.src.search.google_search import google_search_serper
from agents.tools.src.react.tools import answer


def test_react_agent_initialization():
    tools = [google_search_serper, answer]
    task_info = "Test search task"
    agent = ReactAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen-7b-chat",
        task_info=task_info,
        backend="client"
    )
    
    # Check system prompt contains task info and tools
    assert task_info in agent.system_prompt
    assert "google_search" in agent.system_prompt
    assert "answer" in agent.system_prompt
    

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
    assert result_missing["thought"] is None
    assert result_missing["action"] is None
    assert result_missing["input"] is None


def test_react_agent_parse():
    tools = [google_search_serper, answer]
    agent = ReactAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen-7b-chat",
        backend="client"
    )
    
    # Mock the generate method to return a predefined response
    def mock_generate(*args, **kwargs):
        return ["""Thought: I need to search for information.
Action: google_search
Input: {"query": "test query"}"""]
    
    agent.generate = mock_generate
    
    # Test the parse method
    messages_list = [[{"role": "user", "content": "Find information about Python"}]]
    result = agent.parse(messages_list, tools)
    
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "Thought: I need to search for information." in result[0]["content"]
    assert len(result[0]["tool_calls"]) == 1
    assert result[0]["tool_calls"][0]["function"]["name"] == "google_search"
    assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"query": "test query"}' 