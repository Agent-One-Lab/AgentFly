from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import asyncio


@dataclass
class TransformersConfig:
    """Configuration for Transformers backend"""
    temperature: float = 1.0
    max_new_tokens: int = 1024
    trust_remote_code: bool = True
    device_map: str = "auto"


@dataclass
class VLLMConfig:
    """Configuration for VLLM backend"""
    temperature: float = 1.0
    max_new_tokens: int = 1024
    # Add other vLLM specific parameters as needed


@dataclass
class AsyncVLLMConfig:
    """Configuration for Async VLLM backend"""
    temperature: float = 1.0
    max_new_tokens: int = 1024
    # Add other async vLLM specific parameters as needed


@dataclass
class VerlConfig:
    """Configuration for Verl backend"""
    temperature: float = 1.0
    max_new_tokens: int = 1024
    # Add other Verl specific parameters as needed


@dataclass
class AsyncVerlConfig:
    """Configuration for Async Verl backend"""
    temperature: float = 1.0
    max_new_tokens: int = 1024
    # Add other async Verl specific parameters as needed


@dataclass
class ClientConfig:
    """Configuration for Client backend (OpenAI-compatible)"""
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