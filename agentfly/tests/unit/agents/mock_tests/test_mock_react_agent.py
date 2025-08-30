import pytest
from unittest.mock import Mock, patch, AsyncMock
from agentfly.agents.react.react_agent import ReactAgent, parse_react_step, extract_tool_calls, ReactSystemPromptTemplate


class TestMockReactAgent:
    """Test ReactAgent with mocked dependencies for CI environments"""
    
    def test_react_agent_initialization(self, mock_tools):
        """Test ReactAgent initialization without GPU dependencies"""
        tools = [mock_tools["google_search"], mock_tools["answer"]]
        task_info = "Test search task"
        
        agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            task_info=task_info,
            backend="client"  # Use client backend for CI
        )
        
        # Test basic initialization
        assert agent is not None
        assert agent.model_name_or_path == "Qwen/Qwen2.5-3B-Instruct"
        assert agent.template == "qwen2.5"
        assert agent.backend == "client"
        assert len(agent.tools) == 2
        assert agent.max_length == 8192
        
        # Test system prompt contains task info and tools
        assert task_info in agent.system_prompt
        assert "google_search" in agent.system_prompt
        assert "answer" in agent.system_prompt
        assert "ReAct-style agent" in agent.system_prompt
    
    def test_react_agent_system_prompt_formatting(self, mock_tools):
        """Test that ReactAgent system prompt is correctly formatted"""
        tools = [mock_tools["google_search"], mock_tools["answer"]]
        task_info = "Search for information and provide answers"
        
        agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            task_info=task_info,
            backend="client"
        )
        
        # Check system prompt structure
        assert "Think→Act→Observe" in agent.system_prompt
        assert "Thought:" in agent.system_prompt
        assert "Action:" in agent.system_prompt
        assert "Input:" in agent.system_prompt
        assert "Answer:" in agent.system_prompt
        assert task_info in agent.system_prompt
        
        # Check tool schemas are included
        assert "google_search" in agent.system_prompt
        assert "answer" in agent.system_prompt
    
    def test_react_agent_no_task_info(self, mock_tools):
        """Test ReactAgent initialization without task info"""
        tools = [mock_tools["google_search"]]
        
        agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            backend="client"
        )
        
        # Should still have basic system prompt
        assert "ReAct-style agent" in agent.system_prompt
        assert len(agent.tools) == 1
    
    def test_parse_react_step_complete(self):
        """Test parsing complete ReAct step"""
        text = """Thought: I need to find information about Python.
Action: google_search
Input: {"query": "Python programming language"}"""
        
        result = parse_react_step(text)
        
        assert result["thought"] == "I need to find information about Python."
        assert result["action"] == "google_search"
        assert result["input"] == '{"query": "Python programming language"}'
    
    def test_parse_react_step_missing_components(self):
        """Test parsing ReAct step with missing components"""
        text = "Thought: I'm thinking about something."
        result = parse_react_step(text)
        
        assert result["thought"] == "I'm thinking about something."
        assert result["action"] is None
        assert result["input"] is None
    
    def test_parse_react_step_action_only(self):
        """Test parsing ReAct step with only action"""
        text = "Action: search\nInput: {\"query\": \"test\"}"
        result = parse_react_step(text)
        
        assert result["thought"] is None
        assert result["action"] == "search"
        assert result["input"] == '{"query": "test"}'
    
    def test_parse_react_step_case_insensitive(self):
        """Test parsing ReAct step with different case"""
        text = "THOUGHT: I need to think.\nACTION: search\nINPUT: {\"query\": \"test\"}"
        result = parse_react_step(text)
        
        assert result["thought"] == "I need to think."
        assert result["action"] == "search"
        assert result["input"] == '{"query": "test"}'
    
    def test_parse_react_step_multiline_thought(self):
        """Test parsing ReAct step with multiline thought"""
        text = """Thought: I need to think about this
step by step. First, I should consider
the user's request carefully.
Action: search
Input: {"query": "multiline test"}"""
        
        result = parse_react_step(text)
        
        assert "step by step" in result["thought"]
        assert "First, I should consider" in result["thought"]
        assert result["action"] == "search"
        assert result["input"] == '{"query": "multiline test"}'
    
    def test_extract_tool_calls_valid_json(self):
        """Test extracting tool calls from valid JSON input"""
        action_input = '{"name": "google_search", "arguments": {"query": "test"}}'
        result = extract_tool_calls(action_input)
        
        assert len(result) == 1
        assert result[0]["name"] == "google_search"
        assert result[0]["arguments"] == {"query": "test"}
    
    def test_extract_tool_calls_invalid_json(self):
        """Test extracting tool calls from invalid JSON input"""
        action_input = '{"name": "google_search", "arguments": {"query": "test"}'  # Missing }
        result = extract_tool_calls(action_input)
        
        assert len(result) == 0
    
    def test_extract_tool_calls_none_input(self):
        """Test extracting tool calls from None input"""
        result = extract_tool_calls(None)
        assert len(result) == 0
    
    def test_extract_tool_calls_empty_string(self):
        """Test extracting tool calls from empty string"""
        result = extract_tool_calls("")
        assert len(result) == 0
    
    def test_react_agent_parse_single_tool_call(self, mock_tools):
        """Test parsing responses with single tool call"""
        tools = [mock_tools["google_search"], mock_tools["answer"]]
        agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            backend="client"
        )
        
        responses = ["""Thought: I need to search for information.
Action: google_search
Input: {"query": "test query"}"""]
        
        result = agent.parse(responses, tools)
        
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "Thought: I need to search for information." in result[0]["content"][0]["text"]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "google_search"
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"query": "test query"}
        assert result[0]["loss"] is True
    
    def test_react_agent_parse_no_tool_call(self, mock_tools):
        """Test parsing responses with no tool call"""
        tools = [mock_tools["google_search"], mock_tools["answer"]]
        agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            backend="client"
        )
        
        responses = ["Thought: I'm thinking about this problem."]
        
        result = agent.parse(responses, tools)
        
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "Thought: I'm thinking about this problem." in result[0]["content"][0]["text"]
        assert len(result[0]["tool_calls"]) == 0
        assert result[0]["loss"] is True
    
    def test_react_agent_parse_final_answer(self, mock_tools):
        """Test parsing responses with final answer"""
        tools = [mock_tools["google_search"], mock_tools["answer"]]
        agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            backend="client"
        )
        
        responses = ["""Thought: I have enough information now.
Action: answer
Input: {"text": "The answer is 42."}"""]
        
        result = agent.parse(responses, tools)
        
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "The answer is 42." in str(result[0]["tool_calls"][0]["function"]["arguments"])
        assert result[0]["tool_calls"][0]["function"]["name"] == "answer"
    
    def test_react_agent_parse_multiple_responses(self, mock_tools):
        """Test parsing multiple responses"""
        tools = [mock_tools["google_search"], mock_tools["answer"]]
        agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            backend="client"
        )
        
        responses = [
            """Thought: I need to search for information.
Action: google_search
Input: {"query": "first query"}""",
            """Thought: Now I can provide an answer.
Action: answer
Input: {"text": "Final answer"}"""
        ]
        
        result = agent.parse(responses, tools)
        
        assert len(result) == 2
        assert result[0]["tool_calls"][0]["function"]["name"] == "google_search"
        assert result[1]["tool_calls"][0]["function"]["name"] == "answer"
    
    def test_react_agent_tool_schema_validation(self, mock_tools):
        """Test that ReactAgent properly handles tool schemas"""
        tools = [mock_tools["google_search"], mock_tools["answer"]]
        agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            backend="client"
        )
        
        # Verify tools are properly stored
        assert len(agent.tools) == 2
        tool_names = [tool.name for tool in agent.tools]
        assert "google_search" in tool_names
        assert "answer" in tool_names
    
    def test_react_agent_with_mock_llm_engine(self, mock_tools, mock_llm_engine):
        """Test ReactAgent with mocked LLM engine"""
        tools = [mock_tools["google_search"], mock_tools["answer"]]
        
        with patch('agents.agents.agent_base.BaseAgent._setup_backend') as mock_setup:
            mock_setup.return_value = None
            
            agent = ReactAgent(
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
    
    def test_react_agent_chain_generation_integration(self, mock_tools, mock_chain_generation):
        """Test ReactAgent integration with chain generation methods"""
        tools = [mock_tools["google_search"], mock_tools["answer"]]
        
        with patch('agents.agents.agent_base.BaseAgent._setup_backend') as mock_setup:
            mock_setup.return_value = None
            
            agent = ReactAgent(
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
    async def test_react_agent_async_operations(self, mock_tools, mock_llm_engine):
        """Test ReactAgent async operations with mocked dependencies"""
        tools = [mock_tools["google_search"], mock_tools["answer"]]
        
        with patch('agents.agents.agent_base.BaseAgent._setup_backend') as mock_setup:
            mock_setup.return_value = None
            
            agent = ReactAgent(
                "Qwen/Qwen2.5-3B-Instruct",
                tools=tools,
                template="qwen2.5",
                backend="client"
            )
            
            agent.llm_engine = mock_llm_engine
            
            # Mock the generate_async method
            mock_llm_engine.generate_async.return_value = [
                "Thought: I need to search.\nAction: google_search\nInput: {\"query\": \"test\"}"
            ]
            
            # Test that async generation can be called
            result = await agent.llm_engine.generate_async(["test"])
            assert len(result) == 1
            assert "Thought:" in result[0]
            assert "Action:" in result[0]
    
    def test_react_agent_error_handling(self, mock_tools):
        """Test ReactAgent error handling in parsing"""
        tools = [mock_tools["google_search"], mock_tools["answer"]]
        agent = ReactAgent(
            "Qwen/Qwen2.5-3B-Instruct",
            tools=tools,
            template="qwen2.5",
            backend="client"
        )
        
        # Test with malformed JSON in input
        malformed_responses = [
            """Thought: I need to search.
Action: google_search
Input: {"query": "test query"""  # Missing closing }
        ]
        
        result = agent.parse(malformed_responses, tools)
        
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        # Should handle malformed input gracefully
        assert len(result[0]["tool_calls"]) == 0
