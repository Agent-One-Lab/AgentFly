"""Typed events yielded by the rollout generator.

The per-chain loop (:meth:`ChainRollout._run_chain`) is an async generator that
``yield``s these events; :meth:`ChainRollout._run_chains` merges the per-chain streams
into one. Following Claude Code's ``queryLoop`` (which yields a discriminated union of
typed event/message classes rather than one struct with a ``type`` flag), each event is
its own small frozen dataclass and consumers ``isinstance``-dispatch.

``MessageProduced`` is the durable transcript item (Claude Code's ``Message``); the
others are progress signals (Claude Code's ``StreamEvent``). Every event carries
``chain_id`` so a merged, interleaved multi-rollout stream stays unambiguous.

We do **not** stream generation, so there is no token-delta event — one
``MessageProduced`` is emitted per turn once the assistant message is complete.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union


class FinishReason(str, Enum):
    """Shared chain/step stop reasons, distinct from backend generation stop reasons.

    ``str``-Enum: compares and JSON-serializes as the underlying string. ``TERMINAL``
    denotes a model-requested stop; tool-requested and policy-driven stops have
    their own diagnostic labels. The labels do not decide loop control.
    """
    MAX_TURNS = "max_turns"
    NO_TOOL_CALLS = "no_tool_calls"
    REPEATED_NO_TOOL_CALLS = "repeated_no_tool_calls"
    TERMINAL = "terminal"
    MAX_MODEL_LEN = "max_model_len"
    TOOL_CONTROL_END = "tool_control_end"
    INVALID_END = "invalid_end"
    TOOL_ERROR_END = "tool_error_end"
    TOOL_END = "tool_end"


@dataclass(frozen=True)
class ChainStarted:
    chain_id: str
    group_id: Optional[str] = None


@dataclass(frozen=True)
class MessageProduced:
    """One assembled assistant message (durable transcript item)."""

    chain_id: str
    depth: int
    message: dict
    total_token_length: Optional[int] = None


@dataclass(frozen=True)
class ToolObserved:
    """One tool result (progress signal)."""

    chain_id: str
    depth: int
    tool_name: str
    observation: Any
    status: str
    image: Optional[Any] = None


@dataclass(frozen=True)
class ChainEnded:
    """Terminal event for a chain. Lightweight by design — the final ``Chain``/``Step``
    live in ``ChainRollout.chains`` / ``.current_steps`` (the loop writes them as it
    ends), not on the event."""

    chain_id: str
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class RolloutError:
    """A chain raised. Carried as an event so a consumer can surface it without the
    merge deadlocking."""

    chain_id: str
    error: BaseException


RolloutEvent = Union[
    ChainStarted, MessageProduced, ToolObserved, ChainEnded, RolloutError
]
