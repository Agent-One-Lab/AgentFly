from .prompts import (
    INSTRUCTION_DESCRIPTION_PLACEHOLDER,
    InstructionPrompt,
    InstructionSystemPrompt,
    SystemPrompt,
)

import json
import re
from typing import Dict, List

from ...agent_base import BaseAgent
from ....tools.src.shell.tools import run_shell_command

# Match ```mswea_bash_command ... ``` or ```bash ... ``` (non-greedy, DOTALL for multiline command)
BASH_BLOCK_PATTERN = re.compile(
    r"```(?:mswea_bash_command|bash)\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


class BashSWEAgent(BaseAgent):
    """
    Agent for SWE-bench-like tasks using only bash commands.
    Parses responses for a single ```mswea_bash_command ... ``` or ```bash ... ``` block
    and invokes run_shell_command with the extracted command.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tools: List = None,
        system_prompt: str = None,
        **kwargs,
    ):
        _tools = tools if tools is not None else [run_shell_command]
        _system_prompt = system_prompt if system_prompt is not None else InstructionSystemPrompt
        super().__init__(
            model_name_or_path,
            tools=_tools,
            system_prompt=_system_prompt,
            **kwargs,
        )

    @staticmethod
    def _extract_bash_command(response: str) -> str | None:
        """Extract the first bash command from a ```mswea_bash_command ... ``` or ```bash ... ``` block."""
        if not response or not isinstance(response, str):
            return None
        m = BASH_BLOCK_PATTERN.search(response)
        if not m:
            return None
        return m.group(1).strip() or None

    def _parse_single_response(self, response: str) -> Dict:
        """
        Parse a single model response: find one bash block and format as a
        run_shell_command tool call.
        """
        if not response or not isinstance(response, str):
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": response or ""}],
                "tool_calls": [],
                "loss": True,
                "status": "terminal",
            }

        command = self._extract_bash_command(response)
        formatted_tool_calls = []

        if command and (not self.tool_names or "run_shell_command" in self.tool_names):
            formatted_tool_calls.append(
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "run_shell_command",
                        "arguments": json.dumps({"cmd": command}),
                    },
                }
            )

        return {
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
            "tool_calls": formatted_tool_calls,
            "loss": True,
            "status": "continue" if formatted_tool_calls else "terminal",
        }

    def parse(self, responses: List[str]) -> List[Dict]:
        return [self._parse_single_response(r) for r in responses]
