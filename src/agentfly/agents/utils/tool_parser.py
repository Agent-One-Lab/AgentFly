"""Optional vLLM tool-parser integration (ChatCompletionRequest + ToolParserManager)."""

import logging
from typing import Any, Optional

ChatCompletionRequest = None
ToolParserManager = None
VLLM_TOOL_PARSER_AVAILABLE = False

try:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.tool_parsers import ToolParserManager

    VLLM_TOOL_PARSER_AVAILABLE = True
except ImportError:
    try:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager

        VLLM_TOOL_PARSER_AVAILABLE = True
    except ImportError:
        pass


def _silence_tool_parsers() -> None:
    # vLLM has used both namespaces across versions
    prefixes = [
        "vllm.tool_parsers",
        "vllm.entrypoints.openai.tool_parsers",
    ]
    for p in prefixes:
        lg = logging.getLogger(p)
        lg.setLevel(logging.CRITICAL + 1)
        lg.propagate = False


if VLLM_TOOL_PARSER_AVAILABLE:
    _silence_tool_parsers()


def create_tool_parser(tool_parser_name: str, tokenizer: Any) -> Optional[Any]:
    """Return a vLLM tool parser instance for ``tool_parser_name``.

    If ``tokenizer`` is ``None`` (e.g. no local HF tokenizer for the model id),
    returns ``None`` and does not instantiate a parser.

    Raises:
        ImportError: if vLLM tool parsers are not installed or not importable.
    """
    if tokenizer is None:
        return None
    if not VLLM_TOOL_PARSER_AVAILABLE or ToolParserManager is None:
        raise ImportError(
            "vLLM tool parser is not available. Please install vllm to use tool_parser_name."
        )
    parser_cls = ToolParserManager.get_tool_parser(tool_parser_name)
    return parser_cls(tokenizer)
