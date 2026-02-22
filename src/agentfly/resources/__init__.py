"""
Resource engine: decoupled design for managing resources (containers, vLLM)
across backends (local, Slurm, AWS, K8s).

Tools and rewards use the engine to acquire, use (MCP or input-text), and release resources.
"""

from .types import (
    BaseResource,
    ResourceSpec,
    ResourceStatus,
)
from .call_interface import (
    CallInterface,
    MCPToolCall,
    InputTextCall,
)
from .runner import (
    BaseRunner,
    LocalRunner,
    SlurmRunner,
    CloudRunner,
    K8sRunner,
)
from .container_resource import ContainerResource
from .engine import ResourceEngine

__all__ = [
    "BaseResource",
    "ResourceSpec",
    "ResourceStatus",
    "CallInterface",
    "MCPToolCall",
    "InputTextCall",
    "BaseRunner",
    "LocalRunner",
    "SlurmRunner",
    "CloudRunner",
    "K8sRunner",
    "ContainerResource",
    "ResourceEngine",
]
