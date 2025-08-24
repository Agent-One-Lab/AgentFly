import pytest
from unittest.mock import Mock, patch, AsyncMock
from .....agents.auto import AutoAgent
from .....agents.react.react_agent import ReactAgent
from .....agents.specialized.code_agent import CodeAgent


class TestMockAgentIntegration:
    """Integration tests for multiple agents working together with mocked dependencies"""
    
    def test_agent_workflow_code_to_react(self, mock_tools, mock_chain_generation):
        """Test workflow where CodeAgent generates code that ReactAgent uses"""
        # Create CodeAgent
        code_tools = [mock_tools["code_interpreter"]]
        code_agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=code_tools,
            template="qwen-7b-chat",
            backend="client"
        )
        
        # Create ReactAgent
        react_tools = [mock_tools["google_search"], mock_tools["answer"]]
        react_agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=react_tools,
            template="qwen2.5",
            task_info="Use code execution results to provide answers",
            backend="client"
        )
        
        # Test that both agents can be created and configured
        assert isinstance(code_agent, CodeAgent)
        assert isinstance(react_agent, ReactAgent)
        assert len(code_agent.tools) == 1
        assert len(react_agent.tools) == 2
        
        # Test that both agents have the expected methods
        assert hasattr(code_agent, 'parse')
        assert hasattr(react_agent, 'parse')
        assert hasattr(code_agent, 'run_async')
        assert hasattr(react_agent, 'run_async')
    
    def test_agent_workflow_react_to_code(self, mock_tools, mock_chain_generation):
        """Test workflow where ReactAgent decides to use CodeAgent"""
        # Create ReactAgent with code execution capability
        react_tools = [mock_tools["google_search"], mock_tools["code_interpreter"], mock_tools["answer"]]
        react_agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=react_tools,
            template="qwen2.5",
            task_info="Search for information and execute code when needed",
            backend="client"
        )
        
        # Test that ReactAgent can handle code execution tools
        assert len(react_agent.tools) == 3
        tool_names = [tool.name for tool in react_agent.tools]
        assert "google_search" in tool_names
        assert "code_interpreter" in tool_names
        assert "answer" in tool_names
        
        # Test system prompt includes code execution
        assert "code_interpreter" in react_agent.system_prompt
    
    def test_auto_agent_workflow(self, mock_tools, mock_chain_generation):
        """Test AutoAgent creating different agent types in sequence"""
        # Create ReactAgent via AutoAgent
        react_config = {
            "agent_type": "react",
            "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
            "template": "qwen-7b-chat",
            "tools": [mock_tools["google_search"], mock_tools["answer"]],
            "backend": "client"
        }
        
        react_agent = AutoAgent.from_config(react_config)
        assert isinstance(react_agent, ReactAgent)
        
        # Create CodeAgent via AutoAgent
        code_config = {
            "agent_type": "code",
            "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
            "template": "qwen-7b-chat",
            "tools": [mock_tools["code_interpreter"]],
            "backend": "client"
        }
        
        code_agent = AutoAgent.from_config(code_config)
        assert isinstance(code_agent, CodeAgent)
        
        # Test that both agents work independently
        assert react_agent.agent_type != code_agent.agent_type
        assert len(react_agent.tools) != len(code_agent.tools)
    
    def test_agent_tool_sharing(self, mock_tools, mock_chain_generation):
        """Test that agents can share common tools"""
        # Create agents with overlapping tools
        code_agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=[mock_tools["code_interpreter"]],
            template="qwen-7b-chat",
            backend="client"
        )
        
        react_agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=[mock_tools["code_interpreter"], mock_tools["answer"]],
            template="qwen2.5",
            backend="client"
        )
        
        # Test that both agents can use the shared tool
        assert code_agent.tools[0].name == "code_interpreter"
        assert react_agent.tools[0].name == "code_interpreter"
        
        # Test that the tool has the same schema in both agents
        assert code_agent.tools[0].schema == react_agent.tools[0].schema
    
    def test_agent_response_parsing_integration(self, mock_tools, mock_chain_generation):
        """Test that different agents can parse each other's response formats"""
        # Create a response that could come from either agent
        mixed_response = """Thought: I need to calculate something.
Action: code_interpreter
Input: {"code": "print(2 + 2)"}"""
        
        # Test ReactAgent parsing this response
        react_tools = [mock_tools["code_interpreter"], mock_tools["answer"]]
        react_agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=react_tools,
            template="qwen2.5",
            backend="client"
        )
        
        react_result = react_agent.parse([mixed_response], react_tools)
        assert len(react_result) == 1
        assert react_result[0]["role"] == "assistant"
        
        # Test CodeAgent parsing a code-focused response
        code_response = "I'll solve this step by step.\n```python\nx = 2 + 2\nprint(x)\n```"
        code_tools = [mock_tools["code_interpreter"]]
        code_agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=code_tools,
            template="qwen-7b-chat",
            backend="client"
        )
        
        code_result = code_agent.parse([code_response], code_tools)
        assert len(code_result) == 1
        assert code_result[0]["role"] == "assistant"
    
    def test_agent_backend_compatibility(self, mock_tools, mock_chain_generation):
        """Test that agents work with different backends"""
        backends = ["client", "transformers"]
        
        for backend in backends:
            # Test ReactAgent with different backends
            react_agent = ReactAgent(
                "Qwen/Qwen2.5-3B-Instruct",
                tools=[mock_tools["google_search"]],
                template="qwen2.5",
                backend=backend
            )
            assert react_agent.backend == backend
            
            # Test CodeAgent with different backends
            code_agent = CodeAgent(
                "Qwen/Qwen2.5-3B-Instruct",
                tools=[mock_tools["code_interpreter"]],
                template="qwen-7b-chat",
                backend=backend
            )
            assert code_agent.backend == backend
    
    def test_agent_error_handling_integration(self, mock_tools, mock_chain_generation):
        """Test error handling across different agent types"""
        # Test ReactAgent with malformed input
        react_tools = [mock_tools["google_search"]]
        react_agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=react_tools,
            template="qwen2.5",
            backend="client"
        )
        
        malformed_response = "Thought: I need to search.\nAction: google_search\nInput: {invalid json"
        react_result = react_agent.parse([malformed_response], react_tools)
        assert len(react_result) == 1
        assert len(react_result[0]["tool_calls"]) == 0
        
        # Test CodeAgent with malformed input
        code_tools = [mock_tools["code_interpreter"]]
        code_agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=code_tools,
            template="qwen-7b-chat",
            backend="client"
        )
        
        malformed_code_response = "```python\nx = 1\n"  # Missing closing ```
        code_result = code_agent.parse([malformed_code_response], code_tools)
        assert len(code_result) == 1
        assert len(code_result[0]["tool_calls"]) == 0
    
    def test_agent_template_compatibility(self, mock_tools, mock_chain_generation):
        """Test that agents work with different templates"""
        templates = ["qwen-7b-chat", "qwen2.5", "qwen2.5-no-tool"]
        
        for template in templates:
            # Test ReactAgent with different templates
            react_agent = ReactAgent(
                "Qwen/Qwen2.5-3B-Instruct",
                tools=[mock_tools["google_search"]],
                template=template,
                backend="client"
            )
            assert react_agent.template == template
            
            # Test CodeAgent with different templates
            code_agent = CodeAgent(
                "Qwen/Qwen2.5-3B-Instruct",
                tools=[mock_tools["code_interpreter"]],
                template=template,
                backend="client"
            )
            assert code_agent.template == template
    
    def test_agent_async_operations_integration(self, mock_tools, mock_llm_engine):
        """Test async operations across different agent types"""
        # Mock LLM engine for both agents
        with patch('agentfly.agents.agent_base.BaseAgent._setup_backend') as mock_setup:
            mock_setup.return_value = None
            
            # Test ReactAgent async operations
            react_tools = [mock_tools["google_search"]]
            react_agent = ReactAgent(
                "Qwen/Qwen2.5-3B-Instruct",
                tools=react_tools,
                template="qwen2.5",
                backend="client"
            )
            react_agent.llm_engine = mock_llm_engine
            
            # Test CodeAgent async operations
            code_tools = [mock_tools["code_interpreter"]]
            code_agent = CodeAgent(
                "Qwen/Qwen2.5-3B-Instruct",
                tools=code_tools,
                template="qwen-7b-chat",
                backend="client"
            )
            code_agent.llm_engine = mock_llm_engine
            
            # Verify both agents can use the same LLM engine
            assert react_agent.llm_engine is mock_llm_engine
            assert code_agent.llm_engine is mock_llm_engine
    
    def test_agent_system_prompt_integration(self, mock_tools, mock_chain_generation):
        """Test that system prompts are properly integrated across agents"""
        # Test ReactAgent system prompt
        react_tools = [mock_tools["google_search"], mock_tools["answer"]]
        react_agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=react_tools,
            template="qwen2.5",
            task_info="Test task for integration",
            backend="client"
        )
        
        # Test CodeAgent system prompt
        code_tools = [mock_tools["code_interpreter"]]
        code_agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=code_tools,
            template="qwen-7b-chat",
            backend="client"
        )
        
        # Verify both agents have appropriate system prompts
        assert "ReAct-style agent" in react_agent.system_prompt
        assert "multi-turn manner" in code_agent.system_prompt
        assert "Test task for integration" in react_agent.system_prompt
        
        # Verify tool information is included in system prompts
        for tool in react_agent.tools:
            assert tool.name in react_agent.system_prompt
        
        for tool in code_agent.tools:
            assert tool.name in code_agent.system_prompt
    
    def test_agent_chain_generation_integration(self, mock_tools, mock_chain_generation):
        """Test that chain generation methods work across different agent types"""
        # Test ReactAgent chain generation
        react_tools = [mock_tools["google_search"]]
        react_agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=react_tools,
            template="qwen2.5",
            backend="client"
        )
        
        # Test CodeAgent chain generation
        code_tools = [mock_tools["code_interpreter"]]
        code_agent = CodeAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=code_tools,
            template="qwen-7b-chat",
            backend="client"
        )
        
        # Verify both agents have chain generation methods
        for agent in [react_agent, code_agent]:
            assert hasattr(agent, 'run_async')
            assert hasattr(agent, 'get_messages')
            assert hasattr(agent, 'tokenize_trajectories')
            
            # Test that methods can be called (they're mocked)
            messages = agent.get_messages()
            assert isinstance(messages, list)
            
            trajectories = agent.tokenize_trajectories()
            assert isinstance(trajectories, dict)
