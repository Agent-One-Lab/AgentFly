from dataclasses import dataclass
from typing import Optional
from vllm import AsyncEngineArgs


@dataclass(init=False)
class AsyncVLLMConfig:
    """Configuration for Async VLLM backend with engine arguments. Arguments are the same as vLLM's arguments, which can
    be found at https://docs.vllm.ai/en/latest/configuration/engine_args.html. Here listed some important arguments:

    Attributes:
        gpu_memory_utilization (float): The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.
        max_model_len (int): Model context length (prompt and output). If unspecified, will be automatically derived from the model config.
        rope_scaling (dict): Rope scaling. For example, {"rope_type":"dynamic","factor":2.0}.
        trust_remote_code (bool): Whether to trust remote code when loading models.
        pipeline_parallel_size (int): Pipeline parallel size.
        data_parallel_size (int): Data parallel size.
        tensor_parallel_size (int): Tensor parallel size.
    """

    engine_args: AsyncEngineArgs

    def __init__(self, engine_args: Optional[AsyncEngineArgs] = None, **kwargs):
        """Initialize AsyncVLLMConfig.

        Args:
            engine_args: Optional AsyncEngineArgs instance. If provided, kwargs are ignored.
            **kwargs: Arguments to pass to AsyncEngineArgs if engine_args is not provided.
        """
        if engine_args is not None:
            self.engine_args = engine_args
        elif kwargs:
            self.engine_args = AsyncEngineArgs(**kwargs)
        else:
            self.engine_args = AsyncEngineArgs()



@dataclass
class AsyncVerlConfig:
    """Configuration for Async Verl backend.

    Attributes:
    """


@dataclass
class ClientConfig:
    """Configuration for Client backend (OpenAI-compatible)

    This configuration class provides settings for connecting to OpenAI-compatible
    API endpoints, such as local models served via vLLM, Ollama, or other
    compatible servers.

    Attributes:
        base_url: The base URL for the API endpoint. Defaults to localhost:8000.
        max_requests_per_minute: Rate limiting for API requests. Defaults to 100.
        timeout: Request timeout in seconds. Defaults to 600 (10 minutes).
        api_key: API key for authentication. Defaults to "EMPTY" for local servers.
        max_new_tokens: Maximum number of tokens to generate. Defaults to 1024.
        temperature: Sampling temperature for text generation. Defaults to 1.0.
    """

    base_url: str = "http://localhost:8000/v1"
    max_requests_per_minute: int = 100
    timeout: int = 3600
    api_key: str = "EMPTY"

# Backend configuration mapping
BACKEND_CONFIGS = {
    "async_vllm": AsyncVLLMConfig,
    "async_verl": AsyncVerlConfig,
    "client": ClientConfig,
}
