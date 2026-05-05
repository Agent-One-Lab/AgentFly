import inspect
import pytest
from agentfly.utils.llm_backends.llm_backends import AsyncVLLMBackend


@pytest.fixture(scope="session")
def shared_async_vllm_backend():
    """Single shared backend for the whole module to avoid OOM. vLLM does not
    reliably free GPU memory inside the same process, so all tests here go
    through this one engine."""
    return AsyncVLLMBackend(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        template="qwen2.5",
        gpu_memory_utilization=0.7,
        max_model_len=4096,
    )


@pytest.mark.gpu
def test_async_vllm_backend_initialization_defaults(shared_async_vllm_backend):
    """The shared backend wired up the basic attributes."""
    backend = shared_async_vllm_backend

    assert backend.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert backend.template == "qwen2.5"
    assert backend.llm_engine is not None


@pytest.mark.gpu
@pytest.mark.asyncio(scope="session")
async def test_generate_async_single_message(shared_async_vllm_backend):
    """Test async generation with a single message"""
    backend = shared_async_vllm_backend

    messages_list = [[{"role": "user", "content": "Hello, how are you?"}]]
    response = await backend.generate_async(messages_list)

    assert isinstance(response, list)
    assert len(response) == 1
    assert isinstance(response[0], str)
    assert len(response[0]) > 0


@pytest.mark.gpu
@pytest.mark.asyncio(scope="session")
async def test_generate_async_batch_messages(shared_async_vllm_backend):
    """Test async generation with batch messages"""
    backend = shared_async_vllm_backend

    messages_list = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "What is the capital of France?"}],
    ]
    response = await backend.generate_async(messages_list)

    assert isinstance(response, list)
    assert len(response) == 2
    assert all(isinstance(r, str) for r in response)
    assert all(len(r) > 0 for r in response)

@pytest.mark.gpu
@pytest.mark.asyncio(scope="session")
async def test_generate_async_with_custom_temperature(shared_async_vllm_backend):
    """Test async generation with custom temperature"""
    backend = shared_async_vllm_backend

    messages_list = [[{"role": "user", "content": "Tell me a short story."}]]
    response = await backend.generate_async(messages_list, temperature=0.7)

    assert isinstance(response, list)
    assert len(response) == 1
    assert isinstance(response[0], str)
    assert len(response[0]) > 0


@pytest.mark.gpu
@pytest.mark.asyncio(scope="session")
async def test_generate_async_with_custom_max_tokens(shared_async_vllm_backend):
    """Test async generation with custom max_tokens"""
    backend = shared_async_vllm_backend

    messages_list = [[{"role": "user", "content": "Count from 1 to 10."}]]
    response = await backend.generate_async(messages_list, max_tokens=50)

    assert isinstance(response, list)
    assert len(response) == 1
    assert isinstance(response[0], str)
    assert len(response[0]) > 0

@pytest.mark.skip(
    reason="vLLM v1 (v0.19.x) AsyncLLM deadlocks on concurrent identical prompts. "
           "n>1 in AsyncVLLMBackend is currently implemented by duplicating the input "
           "to N copies, which produces N identical concurrent generate() calls and "
           "triggers the deadlock. Re-enable when vLLM upstream fixes this or when "
           "we route n>1 through SamplingParams.n without the v1 generator-exhaustion bug."
)
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_generate_async_with_num_return_sequences(shared_async_vllm_backend):
    """Test async generation with multiple return sequences"""
    backend = shared_async_vllm_backend

    messages_list = [[{"role": "user", "content": "Say hello."}]]
    response = await backend.generate_async(messages_list, n=3)

    print(response)

    assert isinstance(response, list)
    assert len(response) == 3
    assert all(isinstance(r, str) for r in response)
    assert all(len(r) > 0 for r in response)


@pytest.mark.gpu
@pytest.mark.asyncio(scope="session")
async def test_generate_async_with_tools(shared_async_vllm_backend):
    """Test async generation with tools"""
    backend = shared_async_vllm_backend

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


@pytest.mark.gpu
@pytest.mark.asyncio(scope="session")
async def test_generate_streaming(shared_async_vllm_backend):
    """Test streaming generation"""
    backend = shared_async_vllm_backend

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

@pytest.mark.gpu
@pytest.mark.asyncio(scope="session")
async def test_generate_streaming_multiple_messages(shared_async_vllm_backend):
    """Test streaming generation with multiple messages"""
    backend = shared_async_vllm_backend

    messages_list = [
        [{"role": "user", "content": "Say hello."}],
        [{"role": "user", "content": "Say goodbye."}],
    ]
    responses = []
    async for response in backend.generate_streaming(messages_list):
        responses.append(response)

    assert len(responses) > 0
    assert all(isinstance(r, str) for r in responses)

@pytest.mark.gpu
def test_process_inputs_without_vision(shared_async_vllm_backend):
    """Test _process_inputs without vision inputs"""
    backend = shared_async_vllm_backend

    prompts = ["Prompt 1", "Prompt 2"]
    vision_inputs = [[], []]  # Empty vision inputs

    inputs = backend._process_inputs(prompts, vision_inputs)

    assert len(inputs) == 2
    assert inputs[0]["prompt"] == "Prompt 1"
    assert inputs[1]["prompt"] == "Prompt 2"
    assert "multi_modal_data" not in inputs[0]
    assert "multi_modal_data" not in inputs[1]

@pytest.mark.gpu
def test_process_inputs_with_vision(shared_async_vllm_backend):
    """Test _process_inputs with vision inputs"""
    from PIL import Image

    backend = shared_async_vllm_backend

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

@pytest.mark.gpu
def test_apply_chat_template(shared_async_vllm_backend):
    """Test chat template application"""
    backend = shared_async_vllm_backend

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

@pytest.mark.gpu
@pytest.mark.asyncio(scope="session")
async def test_generate_async_batch_with_different_sizes(shared_async_vllm_backend):
    """Test batch generation with different message sizes"""
    backend = shared_async_vllm_backend

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

@pytest.mark.gpu
@pytest.mark.asyncio(scope="session")
async def test_generate_async_with_multi_turn_conversation(shared_async_vllm_backend):
    """Test async generation with multi-turn conversation"""
    backend = shared_async_vllm_backend

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

@pytest.mark.gpu
@pytest.mark.asyncio(scope="session")
async def test_generate_streaming_with_custom_params(shared_async_vllm_backend):
    """Test streaming generation with custom parameters"""
    backend = shared_async_vllm_backend

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
