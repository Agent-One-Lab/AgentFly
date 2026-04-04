import json
import re
from typing import Dict, List, Optional

from ..agent_base import BaseAgent

# Matches both <action>...</action> and <summarize>...</summarize> blocks.
TOOL_TAG_PATTERN = re.compile(
    r"<(action|summarize)>\s*(.*?)\s*</\1>", re.DOTALL | re.IGNORECASE
)
# First closing tag (any case): strip everything after it; keep the tag itself.
CLOSE_TOOL_TAG_PATTERN = re.compile(r"</(action|summarize)>", re.IGNORECASE)


class ActionAgent(BaseAgent):
    """Agent that parses the action format: <think>...</think>, <action> action </action>"""

    def __init__(
        self,
        model_name_or_path: str,
        tool_parser_name: Optional[str] = None,
        tools: List = [],
        **kwargs,
    ):
        self.action_tool_name = tools[0].name
        super().__init__(
            model_name_or_path, tool_parser_name=tool_parser_name, tools=tools, **kwargs
        )

    @staticmethod
    def _truncate_at_first_close_tag(response: str) -> str:
        """Drop any text after the first ``</action>`` or ``</summarize>`` (tag retained)."""
        if not response or not isinstance(response, str):
            return response
        m = CLOSE_TOOL_TAG_PATTERN.search(response)
        return response[: m.end()] if m else response

    def _segment_has_prior_action(self, current_segment: Optional[List[Dict]]) -> bool:
        """True if any prior assistant message in the segment requested the action tool."""
        if not current_segment:
            return False
        for msg in current_segment:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if fn.get("name") == self.action_tool_name:
                    return True
        return False

    def _parse_single_response(self, response: str, current_segment: List[Dict]) -> Dict:
        """
        Parse a single model response using the action format:
        - <think>...</think> (reasoning)
        - <action> action </action> (tool call to action)

        Extracts all <action>...</action> blocks as tool calls; returns one assistant
        message with the same structure as agent_base.parse (content + tool_calls + status).
        If an <action> or <summarize> body equals "end task" after strip (case-insensitive),
        status is "terminal".

        If the parsed tool call is <summarize> and ``current_segment`` has no prior assistant
        tool call to the action tool, status is "terminal" (summarize without any action first).
        """
        # Preprocess: drop anything after the first </action> or </summarize>
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
        end_task_terminal = False
        for i, m in enumerate(TOOL_TAG_PATTERN.finditer(response)):
            tag = m.group(1).lower()
            body = m.group(2).strip()
            if not body:
                continue

            if tag == "action":
                if body.lower() == "end task":
                    end_task_terminal = True
                func_name = self.action_tool_name
                arguments = {"action": body}
            elif tag == "summarize":
                if body.lower() == "end task":
                    end_task_terminal = True
                # Dedicated summarize tool; arguments follow the summarize tool schema.
                func_name = "summarize"
                arguments = {"summary": body}
            else:
                continue

            formatted_tool_calls.append(
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(arguments),
                    },
                }
            )

        # We only support one tool call for now
        if len(formatted_tool_calls) > 1:
            formatted_tool_calls = [formatted_tool_calls[0]]

        summarize_without_prior_action = (
            len(formatted_tool_calls) == 1
            and formatted_tool_calls[0]["function"]["name"] == "summarize"
            and not self._segment_has_prior_action(current_segment)
        )

        return {
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
            "tool_calls": formatted_tool_calls,
            "loss": True,
            "status": "terminal"
            if (
                len(formatted_tool_calls) == 0
                or end_task_terminal
            )
            else "continue",
        }

    def parse(self, responses: List[str], current_segments: List[List[Dict]]) -> List[Dict]:
        return [self._parse_single_response(response, current_segments[i]) for i, response in enumerate(responses)]

    def _segment_is_terminal_summarize_without_prior_action(self, segment: List[Dict]) -> bool:
        """
        True iff this matches the parse-time rule: single summarize tool call and no earlier
        action in the segment (excluding explicit ``summary`` body ``end task``, which is a valid stop).
        """
        if not segment or segment[-1].get("role") != "assistant":
            return False
        last = segment[-1]
        tcs = last.get("tool_calls") or []
        if len(tcs) != 1:
            return False
        fn = (tcs[0].get("function") or {}).get("name")
        if fn != "summarize":
            return False
        if self._segment_has_prior_action(segment[:-1]):
            return False
        raw_args = (tcs[0].get("function") or {}).get("arguments")
        if not raw_args:
            return True
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            summary = (args.get("summary") or "").strip().lower()
            if summary == "end task":
                return False
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return True

    def postprocess_trajectories(self, trajectories: List[Dict]) -> List[Dict]:
        """
        Mark trajectories whose segments contain summarize-without-action shortcuts.
        We keep trajectory/segment shapes unchanged for downstream batch alignment.
        """
        strategy = None
        if strategy is None:
            return trajectories
        elif strategy == "truncate":
            out: List[Dict] = []
            for traj in trajectories:
                traj = dict(traj)
                segs = traj.get("trajectory_segments") or []
                if segs and self._segment_is_terminal_summarize_without_prior_action(segs[-1]):
                    traj["trajectory_segments"] = segs[:-1]
                out.append(traj)
            return out
        elif strategy == "filter":
            out: List[Dict] = []
            for traj in trajectories:
                traj = dict(traj)
                segs = traj.get("trajectory_segments") or []
                has_invalid_summarize_segment = any(
                    self._segment_is_terminal_summarize_without_prior_action(segment)
                    for segment in segs
                )
                traj["discarded"] = has_invalid_summarize_segment
                out.append(traj)
            return out
        raise ValueError(f"Unknown postprocess strategy: {strategy}")
