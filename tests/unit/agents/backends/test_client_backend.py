import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from agentfly.agents.llm_backends.llm_backends import ClientBackend


def test_client_backend_initialization_defaults():
    """Test ClientBackend initialization with default parameters"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    assert backend.model_name == "test-model"
    assert backend.template == "qwen2.5"
    assert backend.base_url == "http://localhost:8000/v1"
    assert backend.timeout == 600
    assert backend.max_new_tokens == 8192
    assert backend._max_tokens == 100
    assert backend.client is not None


def test_client_backend_initialization_custom_params():
    """Test ClientBackend initialization with custom parameters"""
    backend = ClientBackend(
        model_name_or_path="custom-model",
        template="custom-template",
        base_url="https://api.example.com/v1",
        max_requests_per_minute=200,
        timeout=300,
        api_key="custom-key",
        max_length=4096,
        max_new_tokens=2048
    )
    
    assert backend.model_name == "custom-model"
    assert backend.template == "custom-template"
    assert backend.base_url == "https://api.example.com/v1"
    assert backend.timeout == 300
    assert backend.max_length == 4096
    assert backend.max_new_tokens == 2048
    assert backend._max_tokens == 200


@pytest.mark.asyncio
async def test_generate_async_single_message():
    """Test async generation with a single message"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Test response",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Hello"}]
    response = await backend.generate_async(messages)
    
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0] == "Test response"
    backend.client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_generate_async_batch_messages():
    """Test async generation with batch messages"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Response 1",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages_list = [
        [{"role": "user", "content": "Hello 1"}],
        [{"role": "user", "content": "Hello 2"}]
    ]
    response = await backend.generate_async(messages_list)
    
    assert isinstance(response, list)
    assert len(response) == 2
    assert backend.client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_generate_async_with_tools_non_openai():
    """Test async generation with tools for non-OpenAI models"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool"
        }
    }]
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Tool response",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Use the tool"}]
    response = await backend.generate_async(messages, tools=tools)
    
    assert isinstance(response, list)
    # For non-OpenAI models, tool_choice is set to "none", so we get a list of strings
    assert len(response) == 1
    assert response[0] == "Tool response"


@pytest.mark.asyncio
async def test_generate_async_with_openai_model():
    """Test async generation with OpenAI model (gpt in name)"""
    backend = ClientBackend(
        model_name_or_path="gpt-4",
        template="qwen2.5",
        base_url="https://api.openai.com/v1"
    )
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "OpenAI response",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Hello"}]
    response = await backend.generate_async(messages)
    
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0] == "OpenAI response"


@pytest.mark.asyncio
async def test_generate_async_with_num_return_sequences():
    """Test async generation with multiple return sequences"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    # Mock response with 3 choices (n=3)
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {"message": {"content": "Response 1", "tool_calls": None}},
            {"message": {"content": "Response 2", "tool_calls": None}},
            {"message": {"content": "Response 3", "tool_calls": None}}
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Hello"}]
    response = await backend.generate_async(messages, num_return_sequences=3)
    
    assert isinstance(response, list)
    # The response should be flattened from the batch
    assert len(response) == 3
    assert response[0] == "Response 1"
    assert response[1] == "Response 2"
    assert response[2] == "Response 3"
    
    # Verify n parameter was passed correctly
    call_kwargs = backend.client.chat.completions.create.call_args[1]
    assert call_kwargs.get("n") == 3


@pytest.mark.asyncio
async def test_generate_streaming():
    """Test streaming generation"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Streaming response",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Hello"}]
    responses = []
    async for response in backend.generate_streaming(messages):
        responses.append(response)
    
    assert len(responses) == 1
    assert responses[0] == "Streaming response"


def test_generate_sync_single_message():
    """Test synchronous generation with a single message"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Sync response",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Hello"}]
    response = backend.generate(messages)
    
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0] == "Sync response"


def test_apply_chat_template():
    """Test chat template application"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    messages_list = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "Hi"}]
    ]
    
    prompts, vision_inputs = backend.apply_chat_template(
        messages_list, 
        template="qwen2.5",
        add_generation_prompt=True
    )
    
    assert isinstance(prompts, list)
    assert len(prompts) == 2
    assert isinstance(vision_inputs, list)
    assert len(vision_inputs) == 2


def test_convert_to_openai_chat_without_tool_call_processing():
    """Test message conversion without tool call processing"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Hi",
            "tool_calls": [{"id": "call_1", "type": "function"}],
            "tool_call_id": "call_1"
        }
    ]
    
    converted = backend._convert_to_openai_chat_without_tool_call_processing(
        messages, 
        is_openai_model=False
    )
    
    assert "tool_calls" not in converted[1]
    assert "tool_call_id" not in converted[1]


def test_convert_to_openai_chat_with_openai_model():
    """Test message conversion for OpenAI models"""
    backend = ClientBackend(
        model_name_or_path="gpt-4",
        template="qwen2.5"
    )
    
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": None}],
            "tool_calls": [{"id": "call_1", "type": "function"}]
        }
    ]
    
    converted = backend._convert_to_openai_chat_without_tool_call_processing(
        messages,
        is_openai_model=True
    )
    
    # For OpenAI models with tool_calls, content should be removed if it's None
    # The tool_calls should remain for OpenAI models
    assert "tool_calls" in converted[1]
    # Content should be removed if it was None
    if converted[1].get("content") == [{"type": "text", "text": None}]:
        assert "content" not in converted[1]


def test_preprocess_messages_and_args_non_openai():
    """Test message preprocessing for non-OpenAI models"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    messages_list = [
        [{"role": "user", "content": "Hello"}]
    ]
    
    kwargs = {"tools": [{"type": "function", "function": {"name": "test"}}]}
    processed_messages, processed_kwargs = backend._preprocess_messages_and_args(
        messages_list, **kwargs
    )
    
    assert processed_kwargs.get("tool_choice") == "none"


def test_preprocess_messages_and_args_openai():
    """Test message preprocessing for OpenAI models"""
    backend = ClientBackend(
        model_name_or_path="gpt-4",
        template="qwen2.5",
        base_url="https://api.openai.com/v1"
    )
    
    messages_list = [
        [{"role": "user", "content": "Hello"}]
    ]
    
    kwargs = {"tools": [{"type": "function", "function": {"name": "test"}}]}
    processed_messages, processed_kwargs = backend._preprocess_messages_and_args(
        messages_list, **kwargs
    )
    
    assert processed_kwargs.get("tool_choice") == "auto"


@pytest.mark.asyncio
async def test_rate_limiting():
    """Test that rate limiting is applied"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5",
        max_requests_per_minute=2
    )
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Response",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Hello"}]
    
    # Make multiple concurrent requests
    tasks = [backend.generate_async(messages) for _ in range(3)]
    responses = await asyncio.gather(*tasks)
    
    # All requests should complete (rate limiting is per-minute, not blocking)
    assert len(responses) == 3
    assert all(isinstance(r, list) for r in responses)


@pytest.mark.asyncio
async def test_generate_with_custom_kwargs():
    """Test generation with custom kwargs like temperature"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Response",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Hello"}]
    response = await backend.generate_async(
        messages,
        temperature=0.7,
        max_completion_tokens=100
    )
    
    # Verify that custom kwargs were passed to the API
    call_kwargs = backend.client.chat.completions.create.call_args[1]
    assert call_kwargs.get("temperature") == 0.7
    assert call_kwargs.get("max_completion_tokens") == 100


@pytest.mark.asyncio
async def test_generate_with_vision_inputs():
    """Test generation with image inputs"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Image response",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    # Mock the image_to_data_uri function to avoid actual image processing
    with patch('agentfly.agents.llm_backends.llm_backends.image_to_data_uri') as mock_image_uri:
        mock_image_uri.return_value = "data:image/jpeg;base64,test_data"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "image": "test_image_path.jpg"}
                ]
            }
        ]
        
        response = await backend.generate_async(messages)
        
        assert isinstance(response, list)
        assert len(response) == 1
        # Verify that image_to_data_uri was called
        mock_image_uri.assert_called_once_with("test_image_path.jpg")
        
        # Verify that image was converted to image_url format in the API call
        call_kwargs = backend.client.chat.completions.create.call_args[1]
        call_messages = call_kwargs.get("messages", [])
        if call_messages and len(call_messages) > 0:
            content = call_messages[0].get("content", [])
            if isinstance(content, list) and len(content) > 1:
                # Image should be converted to image_url format
                assert content[1].get("type") == "image_url"
                assert "image_url" in content[1]


def test_max_new_tokens_parameter():
    """Test that max_new_tokens parameter is used correctly"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5",
        max_new_tokens=2048
    )
    
    assert backend.max_new_tokens == 2048
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Response",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Hello"}]
    response = backend.generate(messages)
    
    # Verify max_completion_tokens was set
    call_kwargs = backend.client.chat.completions.create.call_args[1]
    assert call_kwargs.get("max_completion_tokens") == 2048


@pytest.mark.asyncio
async def test_generate_batch_with_different_sizes():
    """Test batch generation with different message sizes"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Response",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages_list = [
        [{"role": "user", "content": "Message 1"}],
        [{"role": "user", "content": "Message 2"}, {"role": "assistant", "content": "Response 2"}],
        [{"role": "user", "content": "Message 3"}]
    ]
    
    response = await backend.generate_async(messages_list)
    
    assert isinstance(response, list)
    assert len(response) == 3
    assert backend.client.chat.completions.create.call_count == 3


def test_convert_to_openai_chat_with_image_content():
    """Test message conversion with image content"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    with patch('agentfly.agents.llm_backends.llm_backends.image_to_data_uri') as mock_image_uri:
        mock_image_uri.return_value = "data:image/jpeg;base64,test_data"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "image": "test.jpg"}
                ]
            }
        ]
        
        converted = backend._convert_to_openai_chat_without_tool_call_processing(
            messages,
            is_openai_model=False
        )
        
        assert len(converted) == 1
        assert isinstance(converted[0]["content"], list)
        assert len(converted[0]["content"]) == 2
        assert converted[0]["content"][0]["type"] == "text"
        assert converted[0]["content"][1]["type"] == "image_url"
        assert "image_url" in converted[0]["content"][1]
        mock_image_uri.assert_called_once_with("test.jpg")


@pytest.mark.asyncio
async def test_generate_with_tool_choice_auto():
    """Test generation with tool_choice='auto' for OpenAI models"""
    backend = ClientBackend(
        model_name_or_path="gpt-4",
        template="qwen2.5",
        base_url="https://api.openai.com/v1"
    )
    
    tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool"
        }
    }]
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": "{}"
                            }
                        }
                    ]
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Use the tool"}]
    response = await backend.generate_async(messages, tools=tools)
    
    # For OpenAI models with tools, tool_choice should be "auto"
    call_kwargs = backend.client.chat.completions.create.call_args[1]
    assert call_kwargs.get("tool_choice") == "auto"
    assert call_kwargs.get("tools") == tools


def test_generate_sync_vs_async_context():
    """Test that generate() works in both sync and async contexts"""
    backend = ClientBackend(
        model_name_or_path="test-model",
        template="qwen2.5"
    )
    
    mock_response = Mock()
    mock_response.dict.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Response",
                    "tool_calls": None
                }
            }
        ]
    }
    
    backend.client = Mock()
    backend.client.chat.completions.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Hello"}]
    
    # In sync context, should return list directly
    response = backend.generate(messages)
    assert isinstance(response, list)
    assert len(response) == 1
    
    # In async context, should return a task
    async def test_async():
        task = backend.generate(messages)
        assert isinstance(task, asyncio.Task)
        response = await task
        assert isinstance(response, list)
        assert len(response) == 1
    
    asyncio.run(test_async())
