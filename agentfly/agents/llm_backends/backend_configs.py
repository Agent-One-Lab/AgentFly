from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from vllm import AsyncEngineArgs


@dataclass
class TransformersConfig:
    """Configuration for Transformers backend using Hugging Face models.
    
    Attributes:
        temperature (float): Sampling temperature for text generation. Controls randomness.
            Higher values (e.g., 1.0) make output more random, lower values (e.g., 0.1) make it more deterministic.
            Defaults to 1.0.
        max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 1024.
        trust_remote_code (bool): Whether to trust remote code when loading models.
            This is required for some custom models. Defaults to True.
        device_map (str): Device mapping strategy for model placement.
            Options include "auto", "cpu", "cuda:0", etc. Defaults to "auto".
    """
    temperature: float = 1.0
    max_new_tokens: int = 1024
    trust_remote_code: bool = True
    device_map: str = "auto"


@dataclass
class VLLMConfig:
    """Configuration for VLLM backend for high-performance inference.
    
    Attributes:
        temperature (float): Sampling temperature for text generation. Controls randomness.
            Higher values (e.g., 1.0) make output more random, lower values (e.g., 0.1) make it more deterministic.
            Defaults to 1.0.
        max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 1024.
    """
    temperature: float = 1.0
    max_new_tokens: int = 1024



@dataclass
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
    engine_args: AsyncEngineArgs = AsyncEngineArgs()

    def __init__(self, **kwargs):
        self.engine_args = AsyncEngineArgs(**kwargs)


@dataclass
class VerlConfig:
    """Configuration for Verl backend.
    
    Attributes:
        temperature (float): Sampling temperature for text generation. Controls randomness.
            Higher values (e.g., 1.0) make output more random, lower values (e.g., 0.1) make it more deterministic.
            Defaults to 1.0.
        max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 1024.
    """
    temperature: float = 1.0
    max_new_tokens: int = 1024


@dataclass
class AsyncVerlConfig:
    """Configuration for Async Verl backend.
    
    Attributes:
        temperature (float): Sampling temperature for text generation. Controls randomness.
            Higher values (e.g., 1.0) make output more random, lower values (e.g., 0.1) make it more deterministic.
            Defaults to 1.0.
        max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 1024.
    """
    temperature: float = 1.0
    max_new_tokens: int = 1024


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
    timeout: int = 600
    api_key: str = "EMPTY"
    max_new_tokens: int = 1024
    temperature: float = 1.0


# Backend configuration mapping
BACKEND_CONFIGS = {
    "transformers": TransformersConfig,
    "vllm": VLLMConfig,
    "async_vllm": AsyncVLLMConfig,
    "verl": VerlConfig,
    "async_verl": AsyncVerlConfig,
    "client": ClientConfig,
} 