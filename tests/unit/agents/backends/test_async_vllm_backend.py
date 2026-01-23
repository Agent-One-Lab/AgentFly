import pytest
from agentfly.agents.llm_backends.llm_backends import AsyncVLLMBackend


def test_async_vllm_backend_initialization_defaults():
    """Test AsyncVLLMBackend initialization with default parameters"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    assert backend.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert backend.template == "qwen2.5"
    assert backend.max_tokens == 1024
    assert backend.temperature == 1.0
    assert backend.llm_engine is not None


def test_async_vllm_backend_initialization_custom_params():
    """Test AsyncVLLMBackend initialization with custom parameters"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=2048,
        temperature=0.7,
        gpu_memory_utilization=0.5,
        max_model_len=4096,
    )

    assert backend.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert backend.template == "qwen2.5"
    assert backend.max_tokens == 2048
    assert backend.temperature == 0.7
    assert backend.llm_engine is not None


def test_async_vllm_backend_initialization_with_engine_args():
    """Test AsyncVLLMBackend initialization with pre-configured engine_args"""
    from vllm import AsyncEngineArgs

    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-3B-Instruct", gpu_memory_utilization=0.5
    )

    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
        engine_args=engine_args,
    )

    assert backend.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert backend.llm_engine is not None
    # Verify engine_args.model was set
    assert engine_args.model == "Qwen/Qwen2.5-3B-Instruct"


@pytest.mark.asyncio
async def test_generate_async_single_message():
    """Test async generation with a single message"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    messages_list = [[{"role": "user", "content": "Hello, how are you?"}]]
    response = await backend.generate_async(messages_list)

    assert isinstance(response, list)
    assert len(response) == 1
    assert isinstance(response[0], str)
    assert len(response[0]) > 0


@pytest.mark.asyncio
async def test_generate_async_batch_messages():
    """Test async generation with batch messages"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    messages_list = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "What is the capital of France?"}],
    ]
    response = await backend.generate_async(messages_list)

    assert isinstance(response, list)
    assert len(response) == 2
    assert all(isinstance(r, str) for r in response)
    assert all(len(r) > 0 for r in response)


@pytest.mark.asyncio
async def test_generate_async_with_custom_temperature():
    """Test async generation with custom temperature"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    messages_list = [[{"role": "user", "content": "Tell me a short story."}]]
    response = await backend.generate_async(messages_list, temperature=0.7)

    assert isinstance(response, list)
    assert len(response) == 1
    assert isinstance(response[0], str)
    assert len(response[0]) > 0


@pytest.mark.asyncio
async def test_generate_async_with_custom_max_tokens():
    """Test async generation with custom max_tokens"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    messages_list = [[{"role": "user", "content": "Count from 1 to 10."}]]
    response = await backend.generate_async(messages_list, max_tokens=50)

    assert isinstance(response, list)
    assert len(response) == 1
    assert isinstance(response[0], str)
    assert len(response[0]) > 0


@pytest.mark.asyncio
async def test_generate_async_with_num_return_sequences():
    """Test async generation with multiple return sequences"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    messages_list = [[{"role": "user", "content": "Say hello."}]]
    response = await backend.generate_async(messages_list, n=3)

    print(response)

    assert isinstance(response, list)
    assert len(response) == 3
    assert all(isinstance(r, str) for r in response)
    assert all(len(r) > 0 for r in response)


@pytest.mark.asyncio
async def test_generate_async_with_tools():
    """Test async generation with tools"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"}
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    messages_list = [[{"role": "user", "content": "What's the weather in Beijing?"}]]
    response = await backend.generate_async(messages_list, tools=tools)

    assert isinstance(response, list)
    assert len(response) == 1
    assert isinstance(response[0], str)
    assert len(response[0]) > 0


@pytest.mark.asyncio
async def test_generate_streaming():
    """Test streaming generation"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    messages_list = [[{"role": "user", "content": "Count from 1 to 5."}]]
    responses = []
    async for response in backend.generate_streaming(messages_list):
        responses.append(response)

    assert len(responses) > 0
    # All responses should be strings
    assert all(isinstance(r, str) for r in responses)
    # Concatenated response should have content
    full_response = "".join(responses)
    assert len(full_response) > 0


@pytest.mark.asyncio
async def test_generate_streaming_multiple_messages():
    """Test streaming generation with multiple messages"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    messages_list = [
        [{"role": "user", "content": "Say hello."}],
        [{"role": "user", "content": "Say goodbye."}],
    ]
    responses = []
    async for response in backend.generate_streaming(messages_list):
        responses.append(response)

    assert len(responses) > 0
    assert all(isinstance(r, str) for r in responses)


def test_process_inputs_without_vision():
    """Test _process_inputs without vision inputs"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    prompts = ["Prompt 1", "Prompt 2"]
    vision_inputs = [[], []]  # Empty vision inputs

    inputs = backend._process_inputs(prompts, vision_inputs)

    assert len(inputs) == 2
    assert inputs[0]["prompt"] == "Prompt 1"
    assert inputs[1]["prompt"] == "Prompt 2"
    assert "multi_modal_data" not in inputs[0]
    assert "multi_modal_data" not in inputs[1]


def test_process_inputs_with_vision():
    """Test _process_inputs with vision inputs"""
    from PIL import Image

    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    # Create a simple test image
    test_image = Image.new("RGB", (100, 100), color="red")

    prompts = ["Prompt 1", "Prompt 2"]
    vision_inputs = [[test_image], []]  # First has vision, second doesn't

    inputs = backend._process_inputs(prompts, vision_inputs)

    assert len(inputs) == 2
    assert inputs[0]["prompt"] == "Prompt 1"
    assert "multi_modal_data" in inputs[0]
    assert len(inputs[0]["multi_modal_data"]) == 1
    assert inputs[1]["prompt"] == "Prompt 2"
    assert "multi_modal_data" not in inputs[1]


def test_apply_chat_template():
    """Test chat template application"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    messages_list = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "Hi"}],
    ]

    prompts, vision_inputs = backend.apply_chat_template(
        messages_list, template="qwen2.5", add_generation_prompt=True
    )

    assert isinstance(prompts, list)
    assert len(prompts) == 2
    assert isinstance(vision_inputs, list)
    assert len(vision_inputs) == 2
    assert all(isinstance(p, str) for p in prompts)
    assert all(len(p) > 0 for p in prompts)


@pytest.mark.asyncio
async def test_generate_async_batch_with_different_sizes():
    """Test batch generation with different message sizes"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    messages_list = [
        [{"role": "user", "content": "Message 1"}],
        [
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ],
        [{"role": "user", "content": "Message 3"}],
    ]

    response = await backend.generate_async(messages_list)

    assert isinstance(response, list)
    assert len(response) == 3
    assert all(isinstance(r, str) for r in response)
    assert all(len(r) > 0 for r in response)


@pytest.mark.asyncio
async def test_generate_async_with_multi_turn_conversation():
    """Test async generation with multi-turn conversation"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    messages_list = [
        [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What's my name?"},
        ]
    ]

    response = await backend.generate_async(messages_list)

    assert isinstance(response, list)
    assert len(response) == 1
    assert isinstance(response[0], str)
    assert len(response[0]) > 0
    # The response should mention Alice since it's in the conversation history
    assert "Alice" in response[0] or "alice" in response[0].lower()


@pytest.mark.asyncio
async def test_generate_streaming_with_custom_params():
    """Test streaming generation with custom parameters"""
    backend = AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        max_tokens=1024,
        temperature=1.0,
    )

    messages_list = [[{"role": "user", "content": "List three colors."}]]
    responses = []
    async for response in backend.generate_streaming(
        messages_list, temperature=0.5, max_tokens=100
    ):
        responses.append(response)

    assert len(responses) > 0
    assert all(isinstance(r, str) for r in responses)
    full_response = "".join(responses)
    assert len(full_response) > 0
