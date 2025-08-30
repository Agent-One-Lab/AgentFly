from ....agents.agent_base import BaseAgent
from ....agents.specialized.code_agent import CodeAgent
from ....agents.react.react_agent import ReactAgent
from ....agents.specialized.think_agent import ThinkAgent
from ....tools import code_interpreter, google_search_serper, answer_qa
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["async_vllm", "client"])
def test_agent_initialization_backend(backend: str):
    # Initialize the code agent
    print(f"Testing {backend} backend")
    try:
        tools = [code_interpreter]
        print("Tools initialized")
        agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
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
    assert agent.template == "qwen2.5"
    
    # Test basic methods
    messages = agent.get_messages()
    assert isinstance(messages, list)

@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["async_vllm", "client"])
def test_code_agent_initialization(backend: str):
    tools = [code_interpreter]
    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend=backend
    )
    

@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["async_vllm", "client"])
def test_react_agent_initialization(backend: str):
    tools = [google_search_serper, answer_qa]
    task_info = "Test search task"
    agent = ReactAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        task_info=task_info,
        backend=backend
    )
    
    # Check system prompt contains task info and tools
    assert task_info in agent.system_prompt
    assert "google_search" in agent.system_prompt
    assert "answer" in agent.system_prompt

@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["async_vllm", "client"])
def test_think_agent_initialization(backend: str):
    tools = [code_interpreter]
    agent = ThinkAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend=backend
    )
