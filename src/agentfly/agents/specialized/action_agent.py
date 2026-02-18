import json
import re
from typing import Dict, List, Optional

from ..agent_base import BaseAgent

ACTION_TAG_PATTERN = re.compile(
    r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE
)


class ActionAgent(BaseAgent):
    """Agent that parses the action format: <think>...</think>, <action> action </action>"""

    def __init__(
        self,
        model_name_or_path: str,
        tool_parser_name: Optional[str] = None,
        tools: List = [],
        **kwargs,
    ):
        assert len(tools) == 1, "ActionAgent only supports one tool for now"
        self.action_tool_name = tools[0].name
        super().__init__(
            model_name_or_path, tool_parser_name=tool_parser_name, tools=tools, **kwargs
        )

    @staticmethod
    def _truncate_at_first_close_tag(response: str) -> str:
        """Remove everything after the first </search> or </answer> (inclusive tag kept)."""
        if not response or not isinstance(response, str):
            return response
        end_action = response.find("</action>")
        cut = None
        if end_action >= 0:
            cut = end_action + len("</action>")
        return response[:cut] if cut is not None else response

    def _parse_single_response(self, response: str) -> Dict:
        """
        Parse a single model response using the action format:
        - <think>...</think> (reasoning)
        - <action> action </action> (tool call to action)

        Extracts all <action>...</action> blocks as tool calls; returns one assistant
        message with the same structure as agent_base.parse (content + tool_calls + status).
        """
        # Preprocess: keep only up to (and including) the first </action>
        response = self._truncate_at_first_close_tag(response)

        if not response or not isinstance(response, str):
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": response or ""}],
                "tool_calls": [],
                "loss": True,
                "status": "terminal",
            }
        formatted_tool_calls = []
        for i, m in enumerate(ACTION_TAG_PATTERN.finditer(response)):
            action = m.group(1).strip()
            if not action:
                continue
            formatted_tool_calls.append(
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": self.action_tool_name,
                        "arguments": json.dumps({"action": action}),
                    },
                }
            )

        # We only support one tool call for now
        if len(formatted_tool_calls) > 1:
            formatted_tool_calls = [formatted_tool_calls[0]]

        return {
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
            "tool_calls": formatted_tool_calls,
            "loss": True,
            "status": "continue" if len(formatted_tool_calls) > 0 else "terminal",
        }

    def parse(self, responses: List[str]) -> List[Dict]:
        return [self._parse_single_response(response) for response in responses]
