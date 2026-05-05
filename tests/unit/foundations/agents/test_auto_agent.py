import pytest
from agentfly.agents.auto import AutoAgent
from agentfly.agents import ReactAgent
from agentfly.agents import CodeAgent


def test_auto_agent_from_config_react():
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": None,
        "tools": ["search", "answer_qa"],
        "backend_config": {
            "backend": "client",
        },
    }

    agent = AutoAgent.from_config(config)

    assert isinstance(agent, ReactAgent)
    assert agent.model_name_or_path == "Qwen/Qwen2.5-3B-Instruct"
    assert agent.template == agent.model_name_or_path
    assert len(agent.tools) == 2
    assert agent.backend == "client"


def test_auto_agent_from_config_code():
    config = {
        "agent_type": "code",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": None,
        "tools": ["code_interpreter"],
        "backend_config": {
            "backend": "client",
        },
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
        template=None,
        tools=["search", "answer_qa"],
        debug=True,
        backend_config={
            "backend": "client",
        },
    )

    assert isinstance(agent, ReactAgent)


def test_auto_agent_with_reward():
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": None,
        "tools": ["search", "answer_qa"],
        "reward_name": "qa_f1_reward",
        "backend_config": {
            "backend": "client",
        },
    }

    agent = AutoAgent.from_config(config)

    assert hasattr(agent, "_reward_fn")
    assert agent._reward_fn is not None


def test_auto_agent_invalid_type():
    config = {
        "agent_type": "invalid_type",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": None,
        "tools": ["search", "answer_qa"],
        "backend_config": {
            "backend": "client",
        },
    }

    with pytest.raises(ValueError):
        AutoAgent.from_config(config)


def test_auto_agent_missing_params():
    config = {
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "tools": ["search", "answer_qa"],
        "backend_config": {
            "backend": "client",
        },
    }

    with pytest.raises(ValueError):
        AutoAgent.from_config(config)
