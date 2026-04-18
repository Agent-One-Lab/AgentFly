import json
import re
from typing import Any, Dict, List, Optional

from ...core.context import Context
from ..agent_base import BaseAgent
from ..chain.structures import Node

# Matches both <action>...</action> and <summarize>...</summarize> blocks.
TOOL_TAG_PATTERN = re.compile(
    r"<(action|summarize)>\s*(.*?)\s*</\1>", re.DOTALL | re.IGNORECASE
)
# First closing tag (any case): strip everything after it; keep the tag itself.
CLOSE_TOOL_TAG_PATTERN = re.compile(r"</(action|summarize)>", re.IGNORECASE)

_DEFAULT_CONTEXT_TRIGGER_MESSAGE = (
    "You have reached the configured assistant-turn limit for this segment. "
    "You must now summarize: respond using a single <summarize>...</summarize> block "
    "with a concise summary of progress and key results so far (not <action>)."
)


class ActionAgent(BaseAgent):
    """Agent that parses the action format: <think>...</think>, <action> action </action>.

    With ``context_trigger_turns`` set, when the history contains exactly that many assistant
    turns, a user message is appended before the next generation to force a ``<summarize>`` response.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tool_parser_name: Optional[str] = None,
        tools: List = [],
        context_trigger_turns: Optional[int] = None,
        context_trigger_message: Optional[str] = None,
        **kwargs,
    ):
        self.action_tool_name = tools[0].name
        if context_trigger_turns is not None and context_trigger_turns < 1:
            raise ValueError("context_trigger_turns must be >= 1 when set")
        self.context_trigger_turns = context_trigger_turns
        self.context_trigger_message = (
            context_trigger_message
            if context_trigger_message is not None
            else _DEFAULT_CONTEXT_TRIGGER_MESSAGE
        )
        super().__init__(
            model_name_or_path, tool_parser_name=tool_parser_name, tools=tools, **kwargs
        )

    @staticmethod
    def _count_assistant_turns(turns: List[Dict[str, Any]]) -> int:
        return sum(1 for m in turns if m.get("role") == "assistant")

    @staticmethod
    def _turn_text(turn: Dict[str, Any]) -> str:
        content = turn.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "text":
                return str(first.get("text") or "")
        return ""

    def _maybe_append_context_trigger_user_message(self, current_node: Node) -> None:
        """After ``context_trigger_turns`` assistant messages, append a user nudge to force summarize."""
        if self.context_trigger_turns is None:
            return
        turns = current_node.messages.messages
        if self._count_assistant_turns(turns) != self.context_trigger_turns:
            return
        if turns and turns[-1].get("role") == "user":
            if self._turn_text(turns[-1]).strip() == self.context_trigger_message.strip():
                return
        current_node.messages.add("user", self.context_trigger_message)

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

    def parse(
        self,
        responses: List[str],
        context: Optional[Context] = None,
        **kwargs,
    ) -> List[Dict]:
        trajectory_segments = []
        if context is not None:
            trajectory_segments = context.metadata.get("trajectory_segments", [])
        return [
            self._parse_single_response(
                response,
                trajectory_segments[i] if i < len(trajectory_segments) else [],
            )
            for i, response in enumerate(responses)
        ]

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
        strategy = "truncate_no_assistant"
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
        elif strategy == "truncate_no_assistant":
            out: List[Dict] = []
            for traj in trajectories:
                traj = dict(traj)
                segs = traj.get("trajectory_segments") or []
                if segs:
                    last_seg = segs[-1] or []
                    has_assistant = any(m.get("role") == "assistant" for m in last_seg)
                    if not has_assistant:
                        traj["trajectory_segments"] = segs[:-1]
                out.append(traj)
            return out
        raise ValueError(f"Unknown postprocess strategy: {strategy}")
