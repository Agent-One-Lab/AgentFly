import pytest
from unittest.mock import Mock, patch, AsyncMock
from .....agents.auto import AutoAgent
from .....agents.react.react_agent import ReactAgent
from .....agents.specialized.code_agent import CodeAgent
from .....rewards import qa_f1_reward

def test_auto_agent_registration():
    """Test agent registration functionality"""
    # Test that built-in agents are registered
    assert "react" in AutoAgent.AGENT_MAPPING
    assert "code" in AutoAgent.AGENT_MAPPING
    
    # Test custom agent registration
    class CustomAgent:
        pass
    
    AutoAgent.register_agent("custom", CustomAgent)
    assert "custom" in AutoAgent.AGENT_MAPPING
    assert AutoAgent.AGENT_MAPPING["custom"] == CustomAgent


def test_auto_agent_from_config_react():
    """Test creating ReactAgent from config"""
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
    """Test creating CodeAgent from config"""
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

def test_auto_agent_from_config_with_reward():
    """Test creating agent with reward function"""
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search"],
        "reward_fn": qa_f1_reward,
        "backend": "client"
    }
    
    agent = AutoAgent.from_config(config)
    
    assert isinstance(agent, ReactAgent)

def test_auto_agent_from_pretrained():
    """Test creating agent using from_pretrained method"""
    agent = AutoAgent.from_pretrained(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        agent_type="react",
        template="qwen2.5",
        tools=["google_search", "answer"],
        debug=True,
        backend="client"
    )
    
    assert isinstance(agent, ReactAgent)
    assert agent.model_name_or_path == "Qwen/Qwen2.5-3B-Instruct"
    assert agent.template == "qwen2.5"
    assert agent.backend == "client"

def test_auto_agent_from_config_missing_params():
    """Test config validation with missing parameters"""
    # Missing agent_type
    config1 = {
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search", "answer"],
        "backend": "client"
    }
    
    with pytest.raises(ValueError, match="Missing required parameter"):
        AutoAgent.from_config(config1)
    
    # Missing template
    config2 = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "tools": ["google_search", "answer"],
        "backend": "client"
    }
    
    with pytest.raises(ValueError, match="Missing required parameter"):
        AutoAgent.from_config(config2)
    
    # Missing tools
    config3 = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "backend": "client"
    }
    
    with pytest.raises(ValueError, match="Missing required parameter"):
        AutoAgent.from_config(config3)
    
    # Missing backend
    config4 = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search", "answer"]
    }
    
    with pytest.raises(ValueError, match="Missing required parameter"):
        AutoAgent.from_config(config4)

def test_auto_agent_from_config_invalid_type():
    """Test config validation with invalid agent type"""
    config = {
        "agent_type": "invalid_type",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search", "answer"],
        "backend": "client"
    }
    
    with pytest.raises(ValueError, match="Unknown agent type"):
        AutoAgent.from_config(config)

def test_auto_agent_tool_loading():
    """Test that tools are properly loaded from names"""
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search", "answer"],
        "backend": "client"
    }
    
    agent = AutoAgent.from_config(config)
    assert len(agent.tools) == 2
    assert agent.tools[0].name == "google_search"
    assert agent.tools[1].name == "answer"


def test_auto_agent_debug_mode():
    """Test debug mode configuration"""
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search"],
        "backend": "client",
        "debug": True
    }
    
    agent = AutoAgent.from_config(config)
    assert agent.debug is True

def test_auto_agent_log_file_configuration():
    """Test log file configuration"""
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search"],
        "backend": "client",
        "log_file": "test_agent"
    }
    
    agent = AutoAgent.from_config(config)
    assert hasattr(agent, 'logger')

def test_auto_agent_max_length_configuration():
    """Test max length configuration"""
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search"],
        "backend": "client",
        "max_length": 4096
    }
    
    agent = AutoAgent.from_config(config)
    assert agent.max_length == 4096

def test_auto_agent_task_info_configuration():
    """Test task info configuration for ReactAgent"""
    task_info = "Use web search to find information and provide answers"
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search", "answer"],
        "backend": "client",
        "task_info": task_info
    }
    
    agent = AutoAgent.from_config(config)
    assert isinstance(agent, ReactAgent)
    assert task_info in agent.system_prompt

def test_auto_agent_custom_agent_registration():
    """Test custom agent registration and usage"""
    class CustomTestAgent:
        def __init__(self, **kwargs):
            self.config = kwargs
    
    # Register custom agent
    AutoAgent.register_agent("custom_test", CustomTestAgent)
    
    # Test that it's registered
    assert "custom_test" in AutoAgent.AGENT_MAPPING
    
    # Test creating custom agent
    config = {
        "agent_type": "custom_test",
        "model_name_or_path": "test-model",
        "template": "test-template",
        "tools": ["answer"],
        "backend": "client"
    }
    
    agent = AutoAgent.from_config(config)
    assert isinstance(agent, CustomTestAgent)

def test_auto_agent_error_handling():
    """Test error handling in agent creation"""
    # Test with completely invalid config
    with pytest.raises(ValueError):
        AutoAgent.from_config({})
    
    # Test with None config
    with pytest.raises(ValueError):
        AutoAgent.from_config(None)

def test_auto_agent_environment_specific_config(test_config):
    """Test environment-specific configuration"""
    if test_config["use_mock"]:
        # CI environment - use smaller model and fewer steps
        config = {
            "agent_type": "react",
            "model_name_or_path": test_config["model"],
            "template": "qwen2.5",
            "tools": ["google_search"],
            "backend": test_config["backend"]
        }
        
        agent = AutoAgent.from_config(config)
        assert agent.backend == test_config["backend"]
        assert agent.model_name_or_path == test_config["model"]

def test_auto_agent_tool_validation():
    """Test that tools are properly validated and stored"""
    config = {
        "agent_type": "react",
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "template": "qwen2.5",
        "tools": ["google_search", "answer"],
        "backend": "client"
    }
    
    agent = AutoAgent.from_config(config)
    
    # Verify tools are properly stored
    assert len(agent.tools) == 2
    tool_names = [tool.name for tool in agent.tools]
    assert "google_search" in tool_names
    assert "answer" in tool_names
    
    # Verify tool schemas
    for tool in agent.tools:
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert hasattr(tool, 'schema')
