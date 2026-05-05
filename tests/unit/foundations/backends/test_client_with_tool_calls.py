from agentfly.utils.llm_backends.llm_backends import ClientBackend
import pytest
import openai

@pytest.mark.gpu
@pytest.mark.asyncio
async def test_client_with_tool_calls():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to get the weather of",
                        }
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "Help me to find the weather in Beijing today."}
    ]
    client = ClientBackend(
        model_name_or_path="Qwen/Qwen3-VL-235B-A22B-Instruct",
        base_url="http://localhost:8000/v1",
        max_requests_per_minute=60,
        timeout=300,
        api_key="EMPTY",
    )
    try:
        response = await client.generate(messages, tools=tools)
        print(response)
        assert len(response) == 1
    except openai.APIConnectionError as e:
        pytest.skip(f"Skipping test due to API connection error: {e}")

    
