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

    # Results belong to callers, not an implicit latest-result cache.
    assert not hasattr(agent, "_last_run_result")
    assert not hasattr(agent, "_require_last_run")


def test_base_agent_initializes_without_result_cache(monkeypatch):
    import agentfly.agents.agent_base as agent_base

    # Exercise the real constructor without downloading models or starting a backend.
    monkeypatch.setattr(agent_base, "create_tokenizer", lambda model: object())
    monkeypatch.setattr(agent_base, "create_processor", lambda model: None)
    monkeypatch.setattr(agent_base, "get_jinja_template", lambda template: None)
    monkeypatch.setattr(agent_base.BaseAgent, "_init_llm_engine", lambda *args: object())
    agent = agent_base.BaseAgent(
        "test-model", tools=[], skills=[], backend_config={"backend": "client"}, monitors=[],
    )

    assert not hasattr(agent, "_last_run_result")
    assert not hasattr(agent, "_require_last_run")
    assert not hasattr(agent, "_last_rollout")
    assert not hasattr(agent, "timing_data")


@pytest.mark.parametrize("backend", _BACKENDS)
def test_code_agent_initialization(backend: str):
    tools = [code_interpreter]
    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template=None if backend == "client" else "qwen2.5",
        backend_config={"backend": backend},
    )
