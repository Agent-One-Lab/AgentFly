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
    from ..agent_base import BaseAgent


@dataclass
class Session:
    """One agent's work in an episode. Data only: agents run sessions.

    Attributes:
        id: The session's number in its episode (0 is the main agent).
        parent_id: The session that started this one; ``None`` for session 0.
        agent: The agent that runs this session.
        state: The rollout strategy's loop state (e.g. ``ChainState`` / ``WindowState``),
            which holds the conversation.
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
