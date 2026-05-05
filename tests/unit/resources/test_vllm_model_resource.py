import pytest
from agentfly.resources.models.vllm_model_resource import VLLMModelResource
from agentfly.resources.types import ResourceStatus, VLLMModelResourceSpec


def _build_real_vllm_spec_from_env() -> VLLMModelResourceSpec:
    model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"

    port = 8000
    tp = 1
    pp = 1
    dp = 1
    startup_timeout = 500
    gpu_memory_utilization = 0.8
    tool_call_parser = "hermes"
    template = None

    return VLLMModelResourceSpec(
        model_name_or_path=model_name_or_path,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        data_parallel_size=dp,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        tool_call_parser=tool_call_parser,
        template=template,
    )


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_vllm_model_resource_real_lifecycle_and_generate():
    spec = _build_real_vllm_spec_from_env()
    startup_timeout = 500.0
    resource = VLLMModelResource(spec=spec, startup_timeout=startup_timeout)

    await resource.start()
    try:
        assert resource.resource_id
        assert resource.category == "vllm"
        assert await resource.get_status() == ResourceStatus.RUNNING

        outputs = await resource.generate_async(
            [[{"role": "user", "content": "Reply with exactly: pong"}]],
            max_tokens=16,
            temperature=0.0,
        )
        print(outputs)
        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert isinstance(outputs[0], str)
        assert outputs[0].strip()
    finally:
        await resource.end()
        assert await resource.get_status() == ResourceStatus.STOPPED
