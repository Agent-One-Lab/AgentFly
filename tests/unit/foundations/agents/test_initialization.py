from agentfly.agents import CodeAgent
from agentfly.agents import ReactAgent
from agentfly.tools import code_interpreter, answer_qa
import pytest

_BACKENDS = (
    pytest.param("async_vllm", marks=pytest.mark.gpu),
    "client",
)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_agent_initialization_backend(backend: str):
    # Initialize the code agent
    print(f"Testing {backend} backend")
    try:
        tools = [code_interpreter]
        print("Tools initialized")
        agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template=None if backend == "client" else "qwen2.5",
            backend_config={"backend": backend},
        )
        print("Agent initialized successfully")
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        raise

    # Verify the agent was initialized correctly
    assert agent.backend == backend
    assert agent.tools == tools
    assert agent.model_name_or_path == "Qwen/Qwen2.5-3B-Instruct"

    # No run has been executed yet, so there's no RunResult.
    assert agent._last_run_result is None


@pytest.mark.parametrize("backend", _BACKENDS)
def test_code_agent_initialization(backend: str):
    tools = [code_interpreter]
    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template=None if backend == "client" else "qwen2.5",
        backend_config={"backend": backend},
    )

