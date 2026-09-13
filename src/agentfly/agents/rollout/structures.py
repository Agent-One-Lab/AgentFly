"""Rollout data structures: the per-turn :class:`Step` and its :class:`Chain` container.

A :class:`Step` is the single per-turn record shared by every rollout strategy —
"what the policy saw this turn (``messages``, ending in the action) and what the
environment returned (``tool_result``)." Chain-style rollouts link Steps into a
:class:`Chain` (a growing/branchable list, the tree-search substrate); bounded
step-style rollouts hold a flat list of Steps. The RL signals (``observation`` /
``anchor`` / ``step_reward`` / ``is_action_valid``) are all read through
``tool_result`` so there is exactly one source of truth per turn.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from termcolor import colored

from ...tools.types import ToolResult
from ..utils.messages import Messages


@dataclass
class Step:
    """One turn of interaction: the conversation the policy conditioned on this
    turn (``messages``, ending in the assistant action) plus the environment's
    :class:`~agentfly.tools.types.ToolResult` for that action.

    ``messages`` holds the *cumulative* history for chain-style rollouts and a
    *bounded* window for step-style rollouts — the only thing the two strategies
    differ by. ``tool_result`` is ``None`` for non-tool turns (thoughts, the seed
    step). ``observation`` / ``observation_code`` / ``is_action_valid`` are derived
    from ``tool_result`` so a turn's signals live in one place.
    """

    messages: Messages
    tool_result: Optional[ToolResult] = None
    is_terminal: bool = False
    is_pruned: bool = False
    type: Optional[str] = None
    description: str = ""
    parent: Optional["Step"] = None
    total_token_length: int = 0  # Total token length of the current messages
    children: List["Step"] = field(default_factory=list)

    @property
    def observation(self) -> str:
        """The LLM-facing tool observation for this turn (``""`` when no tool ran)."""
        return self.tool_result.observation if self.tool_result is not None else ""

    @property
    def observation_code(self) -> Optional[str]:
        """The tool status for this turn (``None`` when no tool ran)."""
        return self.tool_result.status if self.tool_result is not None else None

    @property
    def is_action_valid(self) -> float:
        """1.0 if the action was accepted by the env, 0.0 if it was invalid
        (from ``tool_result.metrics['invalid_action']``)."""
        if self.tool_result is None:
            return 1.0
        return 1.0 - float((self.tool_result.metrics or {}).get("invalid_action") or 0.0)

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    def print_step(self, process_id: int = 0) -> None:
        if process_id != 0:
            return
        color_converter = {
            "Thought": "red",
            "Action": "blue",
            "Action Input": "cyan",
            "Final Answer": "green",
            "Reflection": "blue",
        }
        color = color_converter.get(self.type, "white")
        print(colored(f"{self.type}: {self.description}", color=color))
        if self.observation:
            obs = (
                self.observation
                if len(self.observation) < 1536
                else f"{self.observation[:1536]}...(len={len(self.observation)})"
            )
            print(colored(f"Observation: {obs}", color="yellow"))

    def to_json(self, use_messages: bool = False) -> dict:
        json_obj = {
            "is_terminal": self.is_terminal,
            "is_pruned": self.is_pruned,
            "depth": self.depth,
            "type": self.type,
            "description": self.description,
            "messages": self.messages if use_messages else [],
        }
        if self.observation:
            json_obj["observation"] = self.observation
        if self.observation_code is not None:
            json_obj["observation_code"] = self.observation_code
        return json_obj

    def to_json_recursive(self, use_messages: bool = False) -> dict:
        data = self.to_json(use_messages=use_messages)
        data["children"] = [
            child.to_json_recursive(use_messages=use_messages)
            for child in self.children
        ]
        return data


class Chain:
    """
    Manages a sequential chain of steps (chain-of-thought).
    Each step can have at most one child.
    """

    def __init__(self, info: Dict[str, Any]):
        self.root: Optional[Step] = None
        self.info: Dict[str, Any] = info
        self.histories: List[List[Dict[str, Any]]] = []

    def add_step(
        self,
        is_terminal: bool = False,
        is_pruned: bool = False,
        type: Optional[str] = None,
        description: str = "",
        tool_result: Optional[ToolResult] = None,
        messages: Optional[List[Any]] = None,
    ) -> Step:
        messages = Messages.from_turns(messages)
        new_step = Step(
            is_terminal=is_terminal,
            is_pruned=is_pruned,
            type=type,
            description=description,
            tool_result=tool_result,
            messages=messages,
        )
        if self.root is None:
            self.root = new_step
        else:
            current = self.root
            while len(current.children) > 0:
                current = current.children[0]
            current.children = [new_step]
            new_step.parent = current
        return new_step

    def steps(self) -> List[Step]:
        """The chain's steps in order, root → leaf."""
        out: List[Step] = []
        step = self.root
        while step:
            out.append(step)
            step = step.children[0] if step.children else None
        return out

    def to_json(self) -> List[dict]:
        return [step.to_json() for step in self.steps()]
