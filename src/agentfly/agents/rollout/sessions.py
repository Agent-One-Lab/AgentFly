"""Sessions: each agent's work inside one episode.

An episode (:class:`~agentfly.core.context.Context`) keeps its sessions in
``context.sessions``, by id. Session 0 is the episode's main agent; later sessions are
subagents it started. A session can be run several times (for example, a subagent that is
spawned and later asked again), and keeps its agent and its loop state between runs.
"""

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ...tools.types import ToolResult
    from ..agent_base import BaseAgent


@dataclass(kw_only=True)
class LoopState:
    """A rollout strategy's loop state for one session; each strategy extends it
    (``ChainState``, ``WindowState``) and its ``run_loop`` updates it in place.

    Attributes:
        episode_id: The episode the session belongs to.
        info: The session's metadata. For session 0 it is the task's metadata and the
            episode's ids; the reward receives it as keyword arguments and the
            trajectory's metadata comes from it.
        depth: The loop's turn counter, kept across runs.
        have_set_resources: Whether the tools' resources have been set up.
    """

    episode_id: str
    info: Dict[str, Any]
    depth: int = 0
    have_set_resources: bool = False


@dataclass
class SessionResult:
    """Where a session's loop stopped, read from its state (``Rollout.result``).

    Attributes:
        finish_reason: Why the loop stopped; ``None`` before it has run.
        turns: The model's turns (generations) so far.
        last_assistant: The last assistant message, if any.
        last_tool_result: The last tool result, if any.
    """

    finish_reason: Optional[str]
    turns: int
    last_assistant: Optional[Dict[str, Any]] = None
    last_tool_result: Optional["ToolResult"] = None


@dataclass
class Session:
    """One agent's work in an episode. Data only: agents run sessions.

    Attributes:
        id: The session's number in its episode (0 is the main agent).
        parent_id: The session that started this one; ``None`` for session 0.
        agent: The agent that runs this session.
        state: The rollout strategy's loop state (a :class:`LoopState`), which holds
            the conversation.
        spawned_at: The parent session's turn when this session was started.
        info: Free space for tools, e.g. per-session stats.
        task: The running loop when the session runs in the background.
    """

    id: int
    parent_id: Optional[int]
    agent: "BaseAgent"
    state: Any
    spawned_at: Optional[int] = None
    info: Dict[str, Any] = field(default_factory=dict)
    task: Optional[asyncio.Task] = None
