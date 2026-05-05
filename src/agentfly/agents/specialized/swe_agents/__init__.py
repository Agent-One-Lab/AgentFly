from .bash_swe_agent import BashSWEAgent
from .function_call_swe_agent import FunctionCallSWEAgent

# Register the Qwen3 Coder vLLM tool parser so tool_parser_name="qwen3_coder" is available
try:
    from . import qwen3coder_tool_parser_vllm  # noqa: F401
except ImportError:
    pass

from .qwen3coder_swe_agent import Qwen3CoderSWEAgent

__all__ = ["BashSWEAgent", "FunctionCallSWEAgent", "Qwen3CoderSWEAgent"]