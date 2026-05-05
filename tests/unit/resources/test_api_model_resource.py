from __future__ import annotations

import pytest

from agentfly.resources.models.api_model_resource import APIModelResource
from agentfly.resources.types import APIModelResourceSpec, ResourceStatus

# Fill these values manually before running this integration test.
API_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
API_BASE_URL = "http://localhost:8000/v1"
API_HOST = "localhost"
API_PORT = 8000
API_KEY = "EMPTY"
API_STARTUP_TIMEOUT = 60.0


def _build_real_api_model_spec() -> tuple[APIModelResourceSpec, float]:
    model_name_or_path = API_MODEL_NAME
    if not model_name_or_path:
        pytest.skip(
            "Set API_MODEL_NAME in this test file to run the integration test."
        )

    base_url = API_BASE_URL or None
    host = API_HOST or None
    port = API_PORT
    api_key = API_KEY
    startup_timeout = float(API_STARTUP_TIMEOUT)

    if not base_url and not (host and port):
        pytest.skip(
            "Set API_BASE_URL or both API_HOST and API_PORT in this test file."
        )

    spec = APIModelResourceSpec(
        model_name_or_path=model_name_or_path,
        base_url=base_url,
        host=host,
        port=port,
        api_key=api_key,
    )
    return spec, startup_timeout


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_api_model_resource_real_lifecycle_and_generate():
    spec, startup_timeout = _build_real_api_model_spec()
    resource = APIModelResource(spec=spec, startup_timeout=startup_timeout)

    await resource.start()

    assert resource.resource_id
    assert resource.category == "api_model"
    assert await resource.get_status() == ResourceStatus.RUNNING

    print("before generate", flush=True)
    outputs = await resource.generate_async(
        [[{"role": "user", "content": "Reply with exactly: pong"}]],
        max_tokens=16,
        temperature=0.0,
    )
    print(f"outputs: {outputs}")
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)
    assert outputs[0].strip()

    await resource.end()
    assert await resource.get_status() == ResourceStatus.STOPPED
