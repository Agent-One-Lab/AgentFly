"""Optional vLLM tool-parser integration (ChatCompletionRequest + ToolParserManager)."""

import logging
from typing import Any, Optional

# Populated lazily by ensure_vllm_tool_parser() on first use. Importing vllm at
# module load would pull the whole vllm stack into a bare ``import
# agentfly.agents``; consumers must call ensure_vllm_tool_parser() and then read
# these via the module (e.g. ``tool_parser.VLLM_TOOL_PARSER_AVAILABLE``) rather
# than binding them as values at their own import time.
ChatCompletionRequest = None
ToolParserManager = None
VLLM_TOOL_PARSER_AVAILABLE = False
vllm_probed = False


def ensure_vllm_tool_parser() -> bool:
    """Import the optional vLLM tool-parser classes once, on first use.

    Populates the module globals ``ChatCompletionRequest`` /
    ``ToolParserManager`` / ``VLLM_TOOL_PARSER_AVAILABLE`` and returns whether the
    vLLM tool parser is available. Deferred because importing vllm is heavy.
    """
    global ChatCompletionRequest, ToolParserManager, VLLM_TOOL_PARSER_AVAILABLE
    global vllm_probed
    if vllm_probed:
        return VLLM_TOOL_PARSER_AVAILABLE
    vllm_probed = True
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.tool_parsers import ToolParserManager
    except ImportError:
        try:
            from vllm.entrypoints.openai.protocol import ChatCompletionRequest
            from vllm.entrypoints.openai.tool_parsers import ToolParserManager
        except ImportError:
            return False
    globals()["ChatCompletionRequest"] = ChatCompletionRequest
    globals()["ToolParserManager"] = ToolParserManager
    VLLM_TOOL_PARSER_AVAILABLE = True
    _silence_tool_parsers()
    return True


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


def create_tool_parser(tool_parser_name: str, tokenizer: Any) -> Optional[Any]:
    """Return a vLLM tool parser instance for ``tool_parser_name``.

    If ``tokenizer`` is ``None`` (e.g. no local HF tokenizer for the model id),
    returns ``None`` and does not instantiate a parser.

    Raises:
        ImportError: if vLLM tool parsers are not installed or not importable.
    """
    if tokenizer is None:
        return None
    if not ensure_vllm_tool_parser() or ToolParserManager is None:
        raise ImportError(
            "vLLM tool parser is not available. Please install vllm to use tool_parser_name."
        )
    parser_cls = ToolParserManager.get_tool_parser(tool_parser_name)
    return parser_cls(tokenizer)
