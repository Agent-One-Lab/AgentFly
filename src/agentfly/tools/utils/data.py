"""Tool data downloads.

The implementation now lives in the centralized, approval-gated
:mod:`agentfly.utils.download` module so every download (training data and tool
assets alike) shows its size and asks for approval. This thin module re-exports
``ensure_tool_data`` for the existing import path and CLI use.
"""

from ...utils.download import ensure_tool_data

__all__ = ["ensure_tool_data"]


if __name__ == "__main__":
    ensure_tool_data("async_dense_retrieve")
