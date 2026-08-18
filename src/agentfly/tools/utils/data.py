"""Tool data downloads.

The implementation now lives in the centralized, approval-gated
:mod:`agentfly.utils.download` module so every download (training data and tool
assets alike) shows its size and asks for approval. This thin module re-exports
``download_tool_data`` for the existing import path and CLI use.
"""

from ...utils.download import download_tool_data

__all__ = ["download_tool_data"]


if __name__ == "__main__":
    download_tool_data("asyncdense_retrieve")
