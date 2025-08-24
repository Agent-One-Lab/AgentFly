import pytest
from ....agents.auto import AutoAgent
from ....agents.react.react_agent import ReactAgent
from ....agents.specialized.code_agent import CodeAgent


def test_auto_agent_from_config_react():
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search", "answer"],
        "backend": "client"
    }
    
    agent = AutoAgent.from_config(config)
    
    assert isinstance(agent, ReactAgent)
    assert agent.model_name_or_path == "Qwen/Qwen2.5-3B-Instruct"
    assert agent.template == "qwen2.5"
    assert len(agent.tools) == 2
    assert agent.backend == "client"


def test_auto_agent_from_config_code():
    config = {
        "agent_type": "code",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["code_interpreter"],
        "backend": "client"
    }
    
    agent = AutoAgent.from_config(config)
    
    assert isinstance(agent, CodeAgent)
    assert agent.model_name_or_path == "Qwen/Qwen2.5-3B-Instruct"
    assert len(agent.tools) == 1
    assert agent.backend == "client"


def test_auto_agent_from_pretrained():
    agent = AutoAgent.from_pretrained(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        agent_type="react",
        template="qwen2.5",
        tools=["google_search", "answer"],
        debug=True,
        backend="client"
    )
    
    assert isinstance(agent, ReactAgent)

def test_auto_agent_with_reward():
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search", "answer"],
        "reward_name": "qa_f1_reward",
        "backend": "client"
    }
    
    agent = AutoAgent.from_config(config)
    
    assert hasattr(agent, "_reward_fn")
    assert agent._reward_fn is not None


def test_auto_agent_invalid_type():
    config = {
        "agent_type": "invalid_type",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwesn2.5",
        "tools": ["google_search", "answer"],
        "backend": "client"
    }
    
    with pytest.raises(ValueError):
        AutoAgent.from_config(config)


def test_auto_agent_missing_params():
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        # Missing template
        "tools": ["google_search", "answer"],
        "backend": "client"
    }
    
    with pytest.raises(ValueError):
        AutoAgent.from_config(config)