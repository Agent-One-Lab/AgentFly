"""Model resources: :class:`VLLMModelResource` launches a local vLLM server, :class:`APIModelResource` uses an existing OpenAI-compatible endpoint."""

from .api_model_resource import APIModelResource
from .vllm_model_resource import VLLMModelResource

__all__ = ["VLLMModelResource", "APIModelResource"]
