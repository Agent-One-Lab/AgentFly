"""
SWE agent that uses the Qwen3 Coder vLLM tool parser for tool call parsing.

Use this agent when running Qwen3 Coder (or compatible) models with vLLM; the parser
handles <tool_call>...</tool_call> XML-style tool calls. Ensure the qwen3_coder parser
is registered by importing from this package before creating the agent, e.g.:

    from agentfly.agents.specialized.swe_agents import Qwen3CoderSWEAgent
    agent = Qwen3CoderSWEAgent(model_name_or_path="Qwen/...", tools=[...])
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from ....core.context import Context
from ...agent_base import BaseAgent
from ...chain.structures import Node
from ..action_agent import (
    CONTEXT_TRIGGER_MESSAGE_DICT,
    CONTEXT_TRIGGER_MESSAGE_TYPE_ENV,
    CONTEXT_TRIGGER_TURNS_ENV,
)
from ....tools.src.shell.tools import run_shell_command
from .prompts import Qwen3CoderSystemPrompt, Qwen3CoderToolPrompt


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
        context_trigger_turns: Optional[int] = None,
        context_trigger_message_type: Optional[str] = None,
        **kwargs,
    ):
        _tools = tools if tools is not None else [run_shell_command]
        system_prompt = system_prompt if system_prompt is not None else Qwen3CoderToolPrompt

        if context_trigger_turns is None:
            env_turns = os.getenv(CONTEXT_TRIGGER_TURNS_ENV)
            if env_turns is not None and str(env_turns).strip() != "":
                try:
                    context_trigger_turns = int(env_turns)
                except ValueError as exc:
                    raise ValueError(
                        f"{CONTEXT_TRIGGER_TURNS_ENV} must be an integer, got: {env_turns!r}"
                    ) from exc
        if context_trigger_turns is not None and context_trigger_turns < 1:
            raise ValueError("context_trigger_turns must be >= 1 when set")

        self.context_trigger_turns = context_trigger_turns
        message_type = context_trigger_message_type
        if message_type is None:
            message_type = os.getenv(CONTEXT_TRIGGER_MESSAGE_TYPE_ENV, "base")
        self.context_trigger_message = CONTEXT_TRIGGER_MESSAGE_DICT.get(
            message_type, CONTEXT_TRIGGER_MESSAGE_DICT["base"]
        )

        super().__init__(
            model_name_or_path,
            tools=_tools,
            system_prompt=system_prompt,
            tool_parser_name=tool_parser_name,
            **kwargs,
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

    def _is_context_triggered_segment(self, current_segment: Optional[List[Dict]]) -> bool:
        """True when this segment ends with the injected summarize-trigger user message."""
        if not current_segment:
            return False
        last = current_segment[-1]
        if last.get("role") != "user":
            return False
        return self._turn_text(last).strip() == self.context_trigger_message.strip()

    @staticmethod
    def _extract_forced_summary_text(response: str) -> str:
        """Build summary text for forced summarize mode from the entire raw response."""
        text = (response or "").strip()
        if not text:
            return ""

        # Drop chain-of-thought content from thinking models.
        # If </think> exists, only keep the suffix after the last closing tag.
        if re.search(r"</think>", text, re.IGNORECASE):
            text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()

        # Remove any tool-call payloads that may be emitted despite summarize instruction.
        text = re.sub(
            r"<tool_call>\s*.*?\s*</tool_call>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        # If the model wrapped output in <summarize>...</summarize>, keep only inner text.
        match = re.search(
            r"<summarize>\s*(.*?)\s*</summarize>", text, re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()

        # Fallback for malformed wrapping where tags are present but not well-formed.
        if re.search(r"</?summarize>", text, re.IGNORECASE):
            return re.sub(r"</?summarize>", "", text, flags=re.IGNORECASE).strip()

        return text

    def parse(
        self,
        responses: List[str],
        context: Optional[Context] = None,
        **kwargs,
    ) -> List[Dict]:
        trajectory_segments = []
        if context is not None:
            trajectory_segments = context.metadata.get("trajectory_segments", [])

        # Keep default parser behavior and only override context-triggered turns.
        parsed_messages = super().parse(responses, context=context, **kwargs)
        for i, response in enumerate(responses):
            current_segment = trajectory_segments[i] if i < len(trajectory_segments) else []
            if self._is_context_triggered_segment(current_segment):
                summary_text = self._extract_forced_summary_text(response)
                if not summary_text:
                    parsed_messages[i] = {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response or ""}],
                        "tool_calls": [],
                        "loss": True,
                        "status": "terminal",
                    }
                    continue

                parsed_messages[i] = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response or ""}],
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "summarize",
                                "arguments": json.dumps({"summary": summary_text}),
                            },
                        }
                    ],
                    "loss": True,
                    "status": "terminal" if summary_text.lower() == "end task" else "continue",
                }
        return parsed_messages

    def validate_tool_call(self, tool_call):
        """Allow internal summarize pseudo-tool even if not in external tool list."""
        tool_name = tool_call["function"]["name"]
        if tool_name == "summarize":
            return True
        return super().validate_tool_call(tool_call)
    
    async def _execute_tool_call(
        self,
        context,
        tool_call,
        newest_messages,
        chain,
        chain_id,
        depth,
        have_set_resources,
        enable_streaming,
    ):
        """
        Execute tool call, with special handling for internal summarize pseudo-tool.
        """
        tool_name = tool_call["function"]["name"]
        if tool_name == "summarize":
            raw_args = tool_call["function"].get("arguments")
            summary = ""
            try:
                parsed = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                summary = str(parsed.get("summary") or "").strip()
            except (json.JSONDecodeError, TypeError, AttributeError):
                summary = ""
            return {
                "name": "summarize",
                "arguments": raw_args if isinstance(raw_args, str) else json.dumps(raw_args or {}),
                "observation": summary,
                "status": "continue",
            }
        return await super()._execute_tool_call(
            context,
            tool_call,
            newest_messages,
            chain,
            chain_id,
            depth,
            have_set_resources,
            enable_streaming,
        )
