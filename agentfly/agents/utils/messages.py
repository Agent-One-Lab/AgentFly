from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Union
from copy import deepcopy
import numpy as np

Part = Dict[str, Any]          # e.g., {"type": "text", "text": "..."} or {"type": "image", "image": "..."}
Turn = Dict[str, Any]          # {"role": "...", "content": [Part, ...], ...}
MessageDict = Dict[str, Any]   # {"messages": [Turn, ...], <other meta keys>...}


class MessagesValidationError(ValueError):
    pass


class Messages:
    """
    One message item (a dict with key 'messages'). Validates that each turn has:
      - 'role': str
      - 'content': List[Part]
    Each Part must have a 'type'. Built-in validation for 'text' and 'image' types.

    If allow_legacy_content=True, non-list content like a str or single part dict
    will be coerced into a valid list of parts.
    """

    __slots__ = ("_data", "_strict", "_allow_legacy_content")

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        strict: bool = True,
        allow_legacy_content: bool = True,
    ) -> None:
        if not isinstance(data, Mapping):
            raise MessagesValidationError(f"Message must be a dict, got {type(data).__name__}")
        if "messages" not in data:
            raise MessagesValidationError("Message dict must contain a 'messages' key.")

        self._strict = strict
        self._allow_legacy_content = allow_legacy_content
        turns = self._validate_many_turns(data["messages"])
        self._data: MessageDict = dict(data)
        self._data["messages"] = turns


    def _validate_part(self, p: Mapping[str, Any], idx: int | None = None) -> Part:
        if not isinstance(p, Mapping):
            raise MessagesValidationError(
                f"Message content part{'' if idx is None else f' at index {idx}'} must be a dict, got {type(p).__name__}"
            )
        if "type" not in p:
            raise MessagesValidationError(
                f"Message content part{'' if idx is None else f' at index {idx}'} missing 'type': {p}"
            )

        ptype = p["type"]
        if ptype == "text":
            if "text" not in p:
                raise MessagesValidationError("Text part missing 'text' field.")
            if not isinstance(p["text"], str):
                raise MessagesValidationError("Text part 'text' must be a string.")
            return dict(p)

        if ptype == "image":
            # accept either 'image' (e.g., base64) or 'image_url'
            if "image" not in p and "image_url" not in p:
                raise MessagesValidationError("Image part must include 'image' or 'image_url'.")
            return dict(p)

        # Unknown part type: require at least 'type'; pass through
        return dict(p)

    def _coerce_to_parts(self, content: Any) -> List[Part]:
        """Lenient coercion for legacy content representations."""
        # Already a list, validate each part
        if isinstance(content, list):
            return [self._validate_part(p, i) for i, p in enumerate(content)]

        # A plain string, treat as text part
        if isinstance(content, str):
            return [self._validate_part({"type": "text", "text": content}, 0)]

        # Anything else is invalid
        raise MessagesValidationError(
            f"Turn 'content' must be a list of a string or a dictionary with following format {{type: text, text: str}}, but got {type(content).__name__}."
        )

    # ---------- Turn validation ----------
    def _validate_turn(self, t: Mapping[str, Any], idx: int | None = None) -> Turn:
        if not isinstance(t, Mapping):
            raise MessagesValidationError(
                f"Turn{'' if idx is None else f' at index {idx}'} must be a dict, got {type(t).__name__}"
            )
        if "role" not in t:
            raise MessagesValidationError(
                f"Turn{'' if idx is None else f' at index {idx}'} missing 'role'."
            )
        role = t["role"]
        if not isinstance(role, str):
            raise MessagesValidationError("Turn 'role' must be a string.")

        if "content" not in t:
            raise MessagesValidationError("Turn missing 'content'.")
        else:
            content = t["content"]
            if isinstance(content, list):
                content_parts = [self._validate_part(p, i) for i, p in enumerate(content)]
                if len(content_parts) == 0:
                    raise MessagesValidationError(f"Turn {'' if idx is None else f' at index {idx}'} 'content' must contain at least one part.")
            else:
                if self._allow_legacy_content:
                    content_parts = self._coerce_to_parts(content)
                else:
                    raise MessagesValidationError("Turn 'content' must be a string or a list of dictionary with following format {{type: text, text: str}}.")

        # Shallow copy + normalized fields
        out = dict(t)
        out["role"] = role
        out["content"] = content_parts
        return out

    def _validate_many_turns(self, items: Iterable[Mapping[str, Any]]) -> List[Turn]:
        out: List[Turn] = []
        for i, m in enumerate(items):
            out.append(self._validate_turn(m, i))
        return out

    # ---------- Turn mutation API ----------
    def add(self, role: str, content: Any, **extra: Any) -> None:
        """
        Add a single turn. 'content' can be a list of parts, a string (text),
        or a single part dict if allow_legacy_content=True.
        """
        obj: MutableMapping[str, Any] = {"role": role, "content": content}
        if extra:
            obj.update(extra)
        self.append(obj)

    def append(self, turn: Mapping[str, Any]) -> None:
        self._data["messages"].append(self._validate_turn(turn))

    def extend(self, turns: Iterable[Mapping[str, Any]]) -> None:
        self._data["messages"].extend(self._validate_many_turns(turns))

    # ---------- Metadata helpers ----------
    def set_meta(self, key: str, value: Any) -> None:
        if key == "messages":
            raise MessagesValidationError("Use turn methods to modify 'messages'.")
        self._data[key] = value

    def update_meta(self, **kwargs: Any) -> None:
        if "messages" in kwargs:
            raise MessagesValidationError("Use turn methods to modify 'messages'.")
        self._data.update(kwargs)

    # ---------- Accessors ----------
    @property
    def messages(self) -> List[Turn]:
        return list(self._data["messages"])

    @property
    def meta(self) -> Dict[str, Any]:
        return {k: v for k, v in self._data.items() if k != "messages"}

    def to_dict(self) -> MessageDict:
        out = dict(self._data)
        out["messages"] = list(self._data["messages"])
        return out

    def set_system_prompt(self, system_prompt: str, enforce: bool = True) -> None:
        
        assert isinstance(system_prompt, str), "System prompt must be a string."

        if "messages" in self._data:
            if self._data["messages"][0]["role"] == "system":
                if enforce:
                    self._data["messages"][0]["content"] = [{"type": "text", "text": system_prompt}]
                else:
                    raise MessagesValidationError("System prompt already exists.")
            else:
                self._data["messages"].insert(0, {"role": "system", "content": system_prompt})

    def __len__(self) -> int:
        return len(self._data["messages"])

    def __iter__(self):
        return iter(self._data["messages"])

    def __getitem__(self, idx: int) -> Turn:
        return self._data["messages"][idx]

    def __repr__(self) -> str:
        metas = {k: v for k, v in self._data.items() if k != "messages"}
        return f"Message(turns={len(self)}, meta_keys={list(metas.keys())}, strict={self._strict})"

    def copy(self) -> "Messages":
        return Messages(deepcopy(self._data), strict=self._strict, allow_legacy_content=self._allow_legacy_content)

    # ---------- Constructors ----------
    @classmethod
    def from_turns(
        cls,
        turns: Iterable[Mapping[str, Any]],
        *,
        strict: bool = True,
        allow_legacy_content: bool = True,
        **meta: Any,
    ) -> "Messages":
        return cls({"messages": list(turns), **meta}, strict=strict, allow_legacy_content=allow_legacy_content)


class MessagesList:
    """
    A collection of `Message` items (each is a dict with 'messages').

    `from_data(...)` normalizes any of the supported input shapes:
      1) List[Dict] with 'messages'
      2) List[List[Turn]]
      3) Dict with 'messages'
      4) List[Turn]

    `default_meta` is merged only when wrapping raw turn lists (cases 2 & 4).
    """

    __slots__ = ("_items", "_strict", "_allow_legacy_content")

    def __init__(
        self,
        items: Iterable[Union[Messages, Mapping[str, Any]]] | None = None,
        *,
        strict: bool = True,
        allow_legacy_content: bool = True,
    ):
        self._strict = strict
        self._allow_legacy_content = allow_legacy_content
        self._items: List[Messages] = []
        if items:
            self.extend(items)

    @staticmethod
    def _is_sequence(obj: Any) -> bool:
        return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))

    @classmethod
    def from_data(
        cls,
        data: Union[Mapping[str, Any], Sequence[Any], np.ndarray],
        *,
        default_meta: Mapping[str, Any] | None = None,
        strict: bool = True,
        allow_legacy_content: bool = True,
    ) -> "MessagesList":
        if isinstance(data, np.ndarray):
            data = data.tolist()

        default_meta = dict(default_meta or {})
        ms = cls(strict=strict, allow_legacy_content=allow_legacy_content)

        # 3) Dict with 'messages'
        if isinstance(data, Mapping):
            if "messages" not in data:
                raise MessagesValidationError("Dict input must contain a 'messages' key.")
            ms.append(Messages(data, strict=strict, allow_legacy_content=allow_legacy_content))
            return ms

        if not isinstance(data, Sequence):
            raise MessagesValidationError(f"Unsupported input type: {type(data).__name__}")

        seq = list(data)
        if not seq:
            raise MessagesValidationError("Input is an empty list.")

        first = seq[0]

        # 1) List[Dict] each with 'messages'
        if isinstance(first, Mapping) and "messages" in first:
            for i, item in enumerate(seq):
                if not isinstance(item, Mapping) or "messages" not in item:
                    raise MessagesValidationError(
                        f"List appears to be message items but index {i} is not a dict with 'messages'."
                    )
                ms.append(Messages(item, strict=strict, allow_legacy_content=allow_legacy_content))
            return ms

        # 2) List[List[Turn]]  (each inner list is a list of turns)
        if cls._is_sequence(first) and (len(first) == 0 or (isinstance(first[0], Mapping) and "role" in first[0])):
            for inner in seq:
                ms.append(
                    Messages.from_turns(
                        inner,
                        strict=strict,
                        allow_legacy_content=allow_legacy_content,
                        **default_meta,
                    )
                )
            return ms

        # 4) List[Turn] (single item)
        if isinstance(first, Mapping) and "role" in first:
            ms.append(
                Messages.from_turns(
                    seq,
                    strict=strict,
                    allow_legacy_content=allow_legacy_content,
                    **default_meta,
                )
            )
            return ms

        raise MessagesValidationError(
            "Input does not match any accepted format. "
            "Supported: list of dicts with 'messages', list of turn-lists, "
            "single dict with 'messages', or a single list of turn dicts."
        )

    # ---- collection API ----
    def append(self, item: Union[Messages, Mapping[str, Any]]) -> None:
        self._items.append(
            item if isinstance(item, Messages)
            else Messages(item, strict=self._strict, allow_legacy_content=self._allow_legacy_content)
        )

    def append_turns(self, turns: Iterable[Mapping[str, Any]], **meta: Any) -> None:
        self._items.append(
            Messages.from_turns(
                turns,
                strict=self._strict,
                allow_legacy_content=self._allow_legacy_content,
                **meta,
            )
        )

    def extend(self, items: Iterable[Union[Messages, Mapping[str, Any]]]) -> None:
        for it in items:
            self.append(it)

    # ---- accessors ----
    def to_list(self) -> List[MessageDict]:
        return [m.to_dict() for m in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx: int) -> Messages:
        return self._items[idx]

    def __repr__(self) -> str:
        return f"Messages(n_items={len(self)}, strict={self._strict})"


    

