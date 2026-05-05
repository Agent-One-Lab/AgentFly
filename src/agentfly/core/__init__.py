"""
Core module for shared abstractions used across tools, rewards, and agents.

This module contains cross-cutting concerns like execution context and
rollout metadata that are used throughout the agentic RL framework.
"""

from .context import Context
from .context_config import ContextConfig, resolve_resource_backend

__all__ = [
    "Context",
    "ContextConfig",
    "resolve_resource_backend",
]
