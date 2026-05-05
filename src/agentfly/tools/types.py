"""Typed result and return-shape aliases for tools.

User-authored ``@tool`` functions return ``str`` or ``dict``; the framework
normalizes those into a :class:`ToolResult` at the call site (see
``tool_base.BaseTool._format_result``) so downstream chain code reads typed
fields. The :class:`ToolReturn` ``TypedDict`` is purely an annotation alias
to help authors and type checkers — it has no runtime effect.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TypedDict, Union


@dataclass
class ToolResult:
    """Normalized output of a tool call.

    The framework constructs these via :meth:`from_raw` from whatever the
    user's ``@tool`` function returned. ``observation`` is the string the
    LLM sees as the tool's output; ``info`` carries any additional fields
    the tool returned in a dict (everything except ``observation`` and
    ``image``).
    """

    name: str
    arguments: Dict[str, Any]
    observation: str
    status: str = "success"
    info: Dict[str, Any] = field(default_factory=dict)
    image: Optional[str] = None

    @classmethod
    def from_raw(
        cls,
        raw: Union[str, dict, "ToolResult"],
        *,
        name: str,
        arguments: Dict[str, Any],
        status: str = "success",
        max_length: Optional[int] = None,
    ) -> "ToolResult":
        """Normalize a tool function's raw return into a ``ToolResult``.

        Accepts:
        - ``str``: treated as ``observation``; ``info`` is empty.
        - ``dict``: must have an ``observation`` key. ``image`` is extracted
          to its own field; everything else goes into ``info``.
        - ``ToolResult``: returned as-is.
        """
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str):
            obs = raw[:max_length] if max_length is not None else raw
            return cls(
                name=name,
                arguments=arguments,
                observation=obs,
                status=status,
                info={},
            )
        if isinstance(raw, dict):
            if "observation" not in raw:
                raise ValueError(
                    f"observation key required when tool {name!r} returns a dict; "
                    f"got keys: {list(raw.keys())}"
                )
            raw = dict(raw)  # don't mutate caller's dict
            obs = raw.pop("observation")
            if max_length is not None and len(obs) > max_length:
                obs = obs[:max_length] + "...(truncated)"
            image = raw.pop("image", None)
            return cls(
                name=name,
                arguments=arguments,
                observation=obs,
                status=status,
                info=raw,
                image=image,
            )
        raise ValueError(
            f"Tool {name!r} returned {type(raw).__name__}; "
            "expected str, dict, or ToolResult."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Produce the legacy dict shape consumed by chain code today.

        Used during Phase-1 migration so existing readers keep working.
        New code should read the typed fields directly.
        """
        d: Dict[str, Any] = {
            "name": self.name,
            "arguments": self.arguments,
            "observation": self.observation,
            "status": self.status,
            "info": self.info,
        }
        if self.image is not None:
            d["image"] = self.image
        return d


class ToolReturn(TypedDict, total=False):
    """Annotation alias for what a user ``@tool`` function returns.

    Tools may return a bare ``str`` (treated as ``observation``) **or** a
    dict with these keys. ``observation`` is required if a dict is returned;
    any keys beyond ``observation`` and ``image`` flow through to
    :attr:`ToolResult.info`.
    """

    observation: str
    image: str
