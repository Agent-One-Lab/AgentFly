"""
Call interfaces for invoking a resource.

Resources can be called in two ways:
- MCP tool-call: resource exposes an MCP server; caller invokes tools via MCP.
- Input-text: generic input (HTTP body, exec stdin, etc.) for containers or HTTP to vLLM.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional


class CallInterface(abc.ABC):
    """
    Abstract interface for how a consumer (tool/reward) invokes a resource.
    """

    @abc.abstractmethod
    async def call(self, resource: Any, payload: Any, **kwargs: Any) -> Any:
        """
        Invoke the resource with the given payload.

        Args:
            resource: The acquired resource (BaseResource or backend-specific handle).
            payload: Tool-call args (MCP) or input text / request body (input-text).
            **kwargs: Backend-specific options.

        Returns:
            Result of the call (e.g. MCP tool result, HTTP response body).
        """
        raise NotImplementedError


class MCPToolCall(CallInterface):
    """
    Call resource via MCP server (tool-call).
    Used for vLLM or any resource that exposes an MCP server.
    """

    async def call(self, resource: Any, payload: Any, **kwargs: Any) -> Any:
        # TODO: Integrate with MCP client; payload = tool name + arguments
        raise NotImplementedError("MCP tool-call adapter not implemented yet")


class InputTextCall(CallInterface):
    """
    Call resource via input-text (e.g. HTTP POST, exec_run in container).
    Dispatches to resource.run_code(payload) when the resource supports it
    (e.g. ContainerResource with run_code); otherwise subclasses can override.
    """

    async def call(self, resource: Any, payload: Any, **kwargs: Any) -> Any:
        if hasattr(resource, "run_code") and callable(getattr(resource, "run_code")):
            code = payload if isinstance(payload, str) else str(payload)
            return await resource.run_code(code)
        raise NotImplementedError(
            f"Input-text call not supported for resource {type(resource).__name__}; "
            "resource must implement run_code(payload) or use a custom CallInterface."
        )
