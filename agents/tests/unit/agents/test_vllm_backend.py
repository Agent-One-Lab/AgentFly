from agents.agents.llm_backend import AsyncVLLMBackend, VLLMBackend
import pytest

def test_vllm_backend():
    backend = VLLMBackend(model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct", template="qwen2.5-vl")
    messages_list = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ]
    ]
    result = backend.generate(messages_list)
    print(result)


@pytest.mark.asyncio(loop_scope="session")
async def test_async_vllm_backend():
    backend = AsyncVLLMBackend(model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct", template="qwen2.5-vl")
    messages_list = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ]
    ]
    result = await backend.generate_async(messages_list)
    print(result)