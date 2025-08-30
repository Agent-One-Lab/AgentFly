import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List


@pytest.fixture
def mock_llm_engine():
    """Mock LLM engine for testing"""
    mock_engine = Mock()
    mock_engine.generate_async = AsyncMock()
    mock_engine.generate = Mock()
    return mock_engine


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing"""
    mock_tok = Mock()
    mock_tok.encode = Mock(return_value=[1, 2, 3, 4, 5])
    mock_tok.decode = Mock(return_value="Mocked decoded text")
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 1
    return mock_tok


@pytest.fixture
def mock_processor():
    """Mock processor for testing"""
    mock_proc = Mock()
    mock_proc.encode = Mock(return_value={"input_ids": [1, 2, 3, 4, 5]})
    mock_proc.decode = Mock(return_value="Mocked processed text")
    return mock_proc


@pytest.fixture
def mock_tools():
    """Mock tools for testing"""
    mock_code_interpreter = Mock()
    mock_code_interpreter.name = "code_interpreter"
    mock_code_interpreter.description = "Run Python code"
    mock_code_interpreter.schema = {
        "name": "code_interpreter",
        "description": "Run Python code",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    }
    
    mock_answer = Mock()
    mock_answer.name = "answer"
    mock_answer.description = "Provide final answer"
    mock_answer.schema = {
        "name": "answer",
        "description": "Provide final answer",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The answer text"}
            },
            "required": ["text"]
        }
    }
    
    mock_google_search = Mock()
    mock_google_search.name = "google_search"
    mock_google_search.description = "Search the web"
    mock_google_search.schema = {
        "name": "google_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
    
    return {
        "code_interpreter": mock_code_interpreter,
        "answer": mock_answer,
        "google_search": mock_google_search
    }


@pytest.fixture
def mock_responses():
    """Mock model responses for testing"""
    return {
        "code_agent": [
            "I'll solve this math problem step by step.\n```python\n# Calculate the speed\ns = 9 / 4  # 9 km in 4 hours\nprint(f'Speed: {s} km/h')\n```",
            "Now let me calculate the time for s + 0.5 speed.\n```python\nnew_speed = s + 0.5\nnew_time = 9 / new_speed\nprint(f'New time: {new_time} hours')\n```",
            "The walk takes 204 minutes including coffee shop time."
        ],
        "react_agent": [
            "Thought: I need to search for information about Python programming.\nAction: google_search\nInput: {\"query\": \"Python programming language features\"}",
            "Thought: Based on the search results, I can now provide an answer.\nAction: answer\nInput: {\"text\": \"Python is a high-level programming language known for its simplicity and readability.\"}"
        ],
        "think_agent": [
            "Let me think about this step by step.\n\nFirst, I need to understand the problem...\n\nBased on my reasoning, the answer is 42."
        ]
    }


@pytest.fixture
def test_config():
    """Provide test configuration based on environment"""
    if os.environ.get('CI'):
        return {
            "backend": "client",
            "model": "microsoft/DialoGPT-small",  # Smaller CPU-compatible model
            "max_steps": 2,
            "num_chains": 2,
            "use_mock": True
        }
    else:
        return {
            "backend": "async_vllm",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "max_steps": 4,
            "num_chains": 5,
            "use_mock": False
        }


@pytest.fixture
def mock_chain_generation():
    """Mock chain generation methods"""
    with patch('agents.agents.agent_base.ChainGeneration.run_async') as mock_run, \
         patch('agents.agents.agent_base.ChainGeneration.get_messages') as mock_get_messages, \
         patch('agents.agents.agent_base.ChainGeneration.tokenize_trajectories') as mock_tokenize:
        
        mock_run.return_value = None
        mock_get_messages.return_value = [{"role": "assistant", "content": "Mocked response"}]
        mock_tokenize.return_value = {"input_ids": [[1, 2, 3, 4, 5]], "attention_mask": [[1, 1, 1, 1, 1]]}
        
        yield {
            "run_async": mock_run,
            "get_messages": mock_get_messages,
            "tokenize_trajectories": mock_tokenize
        }


@pytest.fixture
def mock_reward_function():
    """Mock reward function for testing"""
    mock_reward = Mock()
    mock_reward.__call__ = Mock(return_value=0.85)
    mock_reward.name = "mock_reward"
    return mock_reward
