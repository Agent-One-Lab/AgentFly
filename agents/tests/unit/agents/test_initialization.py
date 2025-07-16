from agents.agents.agent_base import BaseAgent
from agents.agents.specialized.code_agent import CodeAgent
from agents.tools import code_interpreter
import pytest


@pytest.mark.parametrize("backend", ["vllm", "client"])
def test_agent_initialization_backend(backend: str):
    # Initialize the code agent
    print(f"Testing {backend} backend")
    try:
        tools = [code_interpreter]
        print("Tools initialized")
        agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen-7b-chat",
            backend=backend
        )
        print("Agent initialized successfully")
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        raise
    
    # Verify the agent was initialized correctly
    assert agent.backend == backend
    assert agent.tools == tools
    assert agent.model_name_or_path == "Qwen/Qwen2.5-3B-Instruct"
    assert agent.template == "qwen-7b-chat"
    
    # Test basic methods
    messages = agent.get_messages()
    assert isinstance(messages, list)


def test_code_agent_initialization():
    tools = [code_interpreter]
    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen-7b-chat",
        backend="client"
    )
    
    # Check system prompt is set correctly
    assert "multi-turn manner" in agent.system_prompt
    assert agent.max_length == 8192


    