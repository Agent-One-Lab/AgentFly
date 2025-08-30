import pytest
from unittest.mock import Mock, patch, AsyncMock
from agentfly.agents.specialized.code_agent import CodeAgent, extract_python_code_markdown, CodeAgentSystemPrompt

def test_code_agent_initialization():
    """Test CodeAgent initialization without GPU dependencies"""
    tools = ["code_interpreter"]
        
    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend="client",  # Use client backend for CI
        debug=True
    )
        
    # Test basic initialization
    assert agent is not None
    assert agent.model_name_or_path == "Qwen/Qwen2.5-3B-Instruct"
    assert agent.template == "qwen2.5"
    assert agent.backend == "client"
    assert len(agent.tools) == 1
    assert agent.max_length == 8192
    
    # Test system prompt
    assert "multi-turn manner" in agent.system_prompt
    assert "python code" in agent.system_prompt.lower()
    assert "code interpreter" in agent.system_prompt.lower()
    

def test_extract_python_code_markdown(self):
    """Test Python code extraction from markdown"""
    # Test single code block
    text1 = "Here's some code:\n```python\nprint('Hello')\n```\nThat's it."
    result1 = extract_python_code_markdown(text1)
    assert len(result1) == 1
    assert "print('Hello')" in result1[0]
    
    # Test multiple code blocks
    text2 = "First:\n```python\nx = 1\n```\nSecond:\n```python\ny = 2\n```"
    result2 = extract_python_code_markdown(text2)
    assert len(result2) == 2
    assert "x = 1" in result2[0]
    assert "y = 2" in result2[1]
    
    # Test no code blocks
    text3 = "Just regular text with no code."
    result3 = extract_python_code_markdown(text3)
    assert len(result3) == 0
    
    # Test code block with different spacing
    text4 = "```python\n   x = 42   \n```"
    result4 = extract_python_code_markdown(text4)
    assert len(result4) == 1
    assert "x = 42" in result4[0]

def test_code_agent_parse_single_code_block(self, mock_tools):
    """Test parsing responses with single code blocks"""
    tools = [mock_tools["code_interpreter"]]
    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend="client"
    )
    
    responses = [
        "I'll solve this step by step.\n```python\nx = 9 / 4\nprint(f'Speed: {x} km/h')\n```"
    ]
    
    result = agent.parse(responses, tools)
    
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "I'll solve this step by step" in result[0]["content"][0]["text"]
    assert len(result[0]["tool_calls"]) == 1
    assert result[0]["tool_calls"][0]["function"]["name"] == "code_interpreter"
    assert "x = 9 / 4" in result[0]["tool_calls"][0]["function"]["arguments"]
    assert result[0]["status"] == "continue"
    assert result[0]["loss"] is True

def test_code_agent_parse_no_code_block(self, mock_tools):
    """Test parsing responses with no code blocks"""
    tools = [mock_tools["code_interpreter"]]
    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend="client"
    )
    
    responses = [
        "I'll solve this problem step by step."
    ]
    
    result = agent.parse(responses, tools)
    
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "I'll solve this problem step by step" in result[0]["content"][0]["text"]
    assert len(result[0]["tool_calls"]) == 0
    assert result[0]["status"] == "terminal"
    assert result[0]["loss"] is True


def test_code_agent_parse_multiple_code_blocks(self):
    """Test parsing responses with multiple code blocks (should fail)"""
    tools = ["code_interpreter"]
    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend="client"
    )
    
    responses = [
        "Here's the first step:\n```python\nx = 1\n```\nAnd the second:\n```python\ny = 2\n```"
    ]
    
    result = agent.parse(responses, tools)
    
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert len(result[0]["tool_calls"]) == 0
    assert result[0]["status"] == "terminal"

def test_code_agent_parse_final_answer():
    """Test parsing responses with final answer"""
    tools = ["code_interpreter"]
    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend="client"
    )
    
    responses = [
        "The final answer is <answer>204 minutes</answer>"
    ]
    
    result = agent.parse(responses, tools)
    
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "204 minutes" in result[0]["content"][0]["text"]
    assert len(result[0]["tool_calls"]) == 0
    assert result[0]["status"] == "terminal"

def test_code_agent_with_mock_llm_engine(mock_llm_engine):
    """Test CodeAgent with mocked LLM engine"""
    tools = ["code_interpreter"]
    
    with patch('agentfly.agents.agent_base.BaseAgent._setup_backend') as mock_setup:
        mock_setup.return_value = None
        
        agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            backend="client"
        )
        
        # Mock the LLM engine
        agent.llm_engine = mock_llm_engine
        
        # Test that the agent can be created and configured
        assert agent.llm_engine is not None
        assert hasattr(agent.llm_engine, 'generate_async')

def test_code_agent_tool_schema_validation():
    """Test that CodeAgent properly handles tool schemas"""
    tools = ["code_interpreter"]
    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend="client"
    )
    
    # Verify tool is properly stored
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "code_interpreter"
    assert "Run Python code" in agent.tools[0].description

def test_code_agent_error_handling():
    """Test CodeAgent error handling in parsing"""
    tools = ["code_interpreter"]
    agent = CodeAgent(
        "Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        template="qwen2.5",
        backend="client"
    )
    
    # Test with malformed response
    malformed_responses = [
        "```python\nx = 1\n"  # Missing closing ```
    ]
    
    result = agent.parse(malformed_responses, tools)
    
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert len(result[0]["tool_calls"]) == 0
    assert result[0]["status"] == "terminal"

def test_code_agent_chain_generation_integration(mock_chain_generation):
    """Test CodeAgent integration with chain generation methods"""
    tools = ["code_interpreter"]
    
    with patch('agentfly.agents.agent_base.BaseAgent._setup_backend') as mock_setup:
        mock_setup.return_value = None
        
        agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            backend="client"
        )
        
        # Test that chain generation methods are available
        assert hasattr(agent, 'run_async')
        assert hasattr(agent, 'get_messages')
        assert hasattr(agent, 'tokenize_trajectories')

@pytest.mark.asyncio
async def test_code_agent_async_operations(mock_llm_engine):
    """Test CodeAgent async operations with mocked dependencies"""
    tools = ["code_interpreter"]
    
    with patch('agentfly.agents.agent_base.BaseAgent._setup_backend') as mock_setup:
        mock_setup.return_value = None
        
        agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            backend="client"
        )
        
        agent.llm_engine = mock_llm_engine
        
        # Mock the generate_async method
        mock_llm_engine.generate_async.return_value = [
            "I'll solve this step by step.\n```python\nx = 9 / 4\nprint(f'Speed: {x} km/h')\n```"
        ]
        
        # Test that async generation can be called
        result = await agent.llm_engine.generate_async(["test"])
        assert len(result) == 1
        assert "```python" in result[0]
