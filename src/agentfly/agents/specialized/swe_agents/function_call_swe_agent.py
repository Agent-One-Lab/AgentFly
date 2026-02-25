import json
import re
from typing import Dict, List, Optional
from ...agent_base import BaseAgent

# Match <function=name> ... </function> (non-greedy inner content, DOTALL for newlines)
FUNCTION_BLOCK_PATTERN = re.compile(
    r"<function=([^>\s]+)>\s*(.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE,
)
# Match <parameter=name> value </parameter> inside a function block
PARAMETER_TAG_PATTERN = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)
# Match single <parameter> json_content </parameter> (for JSON-style arguments)
PARAMETER_JSON_PATTERN = re.compile(
    r"<parameter>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


class FunctionCallSWEAgent(BaseAgent):
    """
    Agent for SWE-bench-like tasks. Parses XML-style function calls:
    <function=name><parameter=key>value</parameter>...</function>
    or <function=name><parameter>{ "key": "value" }</parameter></function>.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tools: List = None,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path,
            tools=tools or [],
            **kwargs,
        )

    @staticmethod
    def _truncate_at_last_function_close(response: str) -> str:
        """Keep content up to and including the last </function> to avoid trailing junk."""
        if not response or not isinstance(response, str):
            return response
        last_close = response.rfind("</function>")
        if last_close >= 0:
            return response[: last_close + len("</function>")]
        return response

    @staticmethod
    def _parse_parameters(inner: str) -> Optional[Dict]:
        """
        Parse inner content of a function block into a dict for tool arguments.
        Supports:
        1) Multiple <parameter=name>value</parameter> -> {"name": "value", ...}
        2) Single <parameter>{"key": "value"}</parameter> -> parsed JSON
        """
        inner = (inner or "").strip()
        if not inner:
            return None

        # Try named parameters first
        named = list(PARAMETER_TAG_PATTERN.finditer(inner))
        if named:
            args = {}
            for m in named:
                key = m.group(1).strip()
                val = m.group(2).strip()
                if key:
                    args[key] = val
            return args if args else None

        # Fallback: single <parameter>...</parameter> as JSON
        json_match = PARAMETER_JSON_PATTERN.search(inner)
        if json_match:
            raw = json_match.group(1).strip()
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {"content": raw}

        return None

    def _parse_single_response(self, response: str) -> Dict:
        """
        Parse a single model response using the XML format:
        <function=name><parameter=key>value</parameter></function>

        Returns one assistant message with content, tool_calls, loss, status.
        """
        response = self._truncate_at_last_function_close(response)

        if not response or not isinstance(response, str):
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": response or ""}],
                "tool_calls": [],
                "loss": True,
                "status": "terminal",
            }

        formatted_tool_calls = []
        for i, m in enumerate(FUNCTION_BLOCK_PATTERN.finditer(response)):
            func_name = m.group(1).strip()
            inner = m.group(2)
            if not func_name:
                continue
            # Only allow tools that this agent knows
            if self.tool_names and func_name not in self.tool_names:
                continue
            params = self._parse_parameters(inner)
            if params is None:
                params = {}
            formatted_tool_calls.append(
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(params),
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
