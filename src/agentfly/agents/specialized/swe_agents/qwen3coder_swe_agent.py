"""
SWE agent that uses the Qwen3 Coder vLLM tool parser for tool call parsing.

Use this agent when running Qwen3 Coder (or compatible) models with vLLM; the parser
handles <tool_call>...</tool_call> XML-style tool calls. Ensure the qwen3_coder parser
is registered by importing from this package before creating the agent, e.g.:

    from agentfly.agents.specialized.swe_agents import Qwen3CoderSWEAgent
    agent = Qwen3CoderSWEAgent(model_name_or_path="Qwen/...", tools=[...])
"""

from typing import List, Optional

from ...agent_base import BaseAgent
from ....tools.src.shell.tools import run_shell_command
from .prompts import Qwen3CoderSystemPrompt

class Qwen3CoderSWEAgent(BaseAgent):
    """
    Agent for SWE-bench-like tasks using the Qwen3 Coder vLLM tool parser.
    Parses <tool_call><function=name>...</function></tool_call> from model output
    and invokes the corresponding tools (e.g. run_shell_command).
    """

    def __init__(
        self,
        model_name_or_path: str,
        tools: Optional[List] = None,
        system_prompt: Optional[str] = None,
        tool_parser_name: str = "qwen3_coder",
        **kwargs,
    ):
        _tools = tools if tools is not None else [run_shell_command]
        system_prompt = system_prompt if system_prompt is not None else Qwen3CoderSystemPrompt
        super().__init__(
            model_name_or_path,
            tools=_tools,
            system_prompt=system_prompt,
            tool_parser_name=tool_parser_name,
            **kwargs,
        )
