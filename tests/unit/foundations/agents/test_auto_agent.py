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

    assert hasattr(agent, "reward_fn")
    assert agent.reward_fn is not None


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


def _hf_config(**overrides):
    config = {
        "agent_type": "hf",
        "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "backend_config": {"backend": "client", "base_url": "http://localhost:1/v1", "api_key": "EMPTY"},
        "monitors": [],
    }
    config.update(overrides)
    return config


def test_from_config_keeps_a_passed_reward_fn():
    from agentfly.rewards import reward

    @reward(name="auto_agent_test_reward")
    def my_reward(final_response: str) -> float:
        return 1.0

    agent = AutoAgent.from_config(_hf_config(reward_fn=my_reward))
    assert agent.reward_fn is my_reward


def test_from_config_rejects_reward_fn_and_reward_name_together():
    with pytest.raises(ValueError, match="either reward_fn or reward_name"):
        AutoAgent.from_config(_hf_config(reward_fn=lambda **_: 0.0, reward_name="qa_f1_reward"))


def test_from_config_accepts_tool_objects_names_and_no_tools():
    from agentfly.tools import calculator

    agent = AutoAgent.from_config(_hf_config(tools=[calculator, "answer_qa"]))
    assert agent.tool_names == ["calculator", "answer_qa"]
    assert AutoAgent.from_config(_hf_config()).tools == []


def test_from_pretrained_keeps_reward_fn_and_tool_objects():
    from agentfly.tools import calculator

    def my_reward(final_response: str) -> float:
        return 0.5

    agent = AutoAgent.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", agent_type="hf", template=None,
        tools=[calculator], reward_fn=my_reward,
        backend_config={"backend": "client", "base_url": "http://localhost:1/v1", "api_key": "EMPTY"},
        monitors=[],
    )
    assert agent.reward_fn is my_reward and agent.tool_names == ["calculator"]
