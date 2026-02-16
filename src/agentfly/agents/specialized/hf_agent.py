import json
import logging
import re
from typing import Dict, List, Optional

from ..agent_base import BaseAgent

logger = logging.getLogger(__file__)

# Output format from system prompt: <think>...</think>, <search> query </search>, <information>...</information>, <answer>...</answer>
_SEARCH_TAG_PATTERN = re.compile(
    r"<search>\s*(.*?)\s*</search>", re.DOTALL | re.IGNORECASE
)


class HFAgent(BaseAgent):
    def __init__(
        self,
        model_name_or_path: str,
        tool_parser_name: Optional[str] = "hermes",
        **kwargs,
    ):
        super().__init__(
            model_name_or_path, tool_parser_name=tool_parser_name, **kwargs
        )


class SearchR1Agent(BaseAgent):
    """Agent that parses the search-R1 format: <think>, <search> query </search>, <answer>.</answer>"""

    def __init__(
        self,
        model_name_or_path: str,
        tool_parser_name: Optional[str] = None,
        search_tool_name: str = "async_dense_retrieve_api",
        **kwargs,
    ):
        super().__init__(
            model_name_or_path, tool_parser_name=tool_parser_name, **kwargs
        )
        self.search_tool_name = search_tool_name

    @staticmethod
    def _truncate_at_first_close_tag(response: str) -> str:
        """Remove everything after the first </search> or </answer> (inclusive tag kept)."""
        if not response or not isinstance(response, str):
            return response
        end_search = response.find("</search>")
        end_answer = response.find("</answer>")
        cut = None
        if end_search >= 0:
            cut = end_search + len("</search>")
        if end_answer >= 0:
            candidate = end_answer + len("</answer>")
            cut = candidate if cut is None else min(cut, candidate)
        return response[:cut] if cut is not None else response

    def _parse_single_response(self, response: str) -> Dict:
        """
        Parse a single model response using the search-R1 format:
        - <think>...</think> (reasoning)
        - <search> query </search> (tool call to asyncdense_retrieve)
        - <answer>...</answer> (final answer)

        Extracts all <search>...</search> blocks as tool calls; returns one assistant
        message with the same structure as agent_base.parse (content + tool_calls + status).
        """
        # Preprocess: keep only up to (and including) the first </search> or </answer>
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
        for i, m in enumerate(_SEARCH_TAG_PATTERN.finditer(response)):
            query = m.group(1).strip()
            if not query:
                continue
            formatted_tool_calls.append(
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": self.search_tool_name,
                        "arguments": json.dumps({"query": query}),
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
