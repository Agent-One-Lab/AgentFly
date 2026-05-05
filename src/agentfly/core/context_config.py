"""
Options for :class:`~agentfly.core.context.Context`, typically passed from ``agent.run(..., context_config=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class ContextConfig:
    """
    Rollout-scoped settings for resource acquisition (engine backend).

    Pass via ``BaseAgent.run(..., context_config=...)``; forwarded to each chain's ``Context``.
    """

    resource_backend: str = "local"
    """``ResourceEngine`` backend name (e.g. ``\"local\"``, ``\"ray\"``)."""


def resolve_resource_backend(
    metadata: Mapping[str, Any],
    context_config: Optional[ContextConfig] = None,
) -> str:
    """
    Backend for ``ResourceEngine.acquire``.

    Order: non-default ``context_config.resource_backend`` wins; else ``metadata['resource_backend']``;
    else ``context_config.resource_backend`` or ``\"local\"``.
    """
    if context_config is not None:
        rb = str(context_config.resource_backend or "local").strip()
        if rb != "local":
            return rb
    m = metadata.get("resource_backend")
    if m is not None and str(m).strip():
        return str(m).strip()
    if context_config is not None:
        return str(context_config.resource_backend or "local").strip()
    return "local"

