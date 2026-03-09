"""Chain data structures: Node and Chain."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from termcolor import colored

from ..utils.messages import Messages


@dataclass
class Node:
    messages: Messages
    is_terminal: bool = False
    is_pruned: bool = False
    type: Optional[str] = None
    description: str = ""
    observation: str = ""
    observation_code: Optional[str] = None
    parent: Optional["Node"] = None
    total_token_length: int = 0  # Total token length of the current messages
    children: List["Node"] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    def print_node(self, process_id: int = 0) -> None:
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
    Manages a sequential chain of nodes (chain-of-thought).
    Each node can have at most one child.
    """

    def __init__(self, info: Dict[str, Any]):
        self.root: Optional[Node] = None
        self.info: Dict[str, Any] = info

    def add_node(
        self,
        is_terminal: bool = False,
        is_pruned: bool = False,
        type: Optional[str] = None,
        description: str = "",
        observation: str = "",
        observation_code: Optional[str] = None,
        messages: Optional[List[Any]] = None,
    ) -> Node:
        messages = Messages.from_turns(messages)
        new_node = Node(
            is_terminal=is_terminal,
            is_pruned=is_pruned,
            type=type,
            description=description,
            observation=observation,
            observation_code=observation_code,
            messages=messages,
        )
        if self.root is None:
            self.root = new_node
        else:
            current = self.root
            while len(current.children) > 0:
                current = current.children[0]
            current.children = [new_node]
            new_node.parent = current
        return new_node

    def to_json(self) -> List[dict]:
        chain_json = []
        node = self.root
        while node:
            chain_json.append(node.to_json())
            if node.children:
                node = node.children[0]
            else:
                break
        return chain_json
