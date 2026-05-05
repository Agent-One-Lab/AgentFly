"""Typed data classes for agent rollouts.

The framework moves data between agent, chain, tools, rewards, and trainer
through these classes. User-authored code (`@tool`, `@reward` functions) does
not need to construct them — the framework normalizes user returns at the
boundary via the `from_raw` factories on `tools.types.ToolResult` and
`rewards.types.RewardResult`.

Two classes are exposed here:

- :class:`Trajectory` — one rollout's worth of data (conversation segments
  plus its reward signal, metrics, identifiers, and free-form metadata).
- :class:`RunResult` — what `agent.run(...)` returns: a list of trajectories
  with convenience accessors.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class Trajectory(BaseModel):
    """A single rollout's data.

    ``segments`` is the canonical store of conversation data. For a non-folded
    rollout (the common case), ``segments`` contains exactly one segment equal
    to the full conversation. For rollouts that use context folding (e.g. the
    ``summarize`` tool), ``segments`` preserves each pre-fold view. Segments
    are **not** a strict partition: concatenating them does not in general
    reproduce the original conversation. Consumers that want training data
    should iterate ``segments`` rather than treat them as slices.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Conversation data
    segments: List[List[Dict[str, Any]]]

    # RL signals
    reward: Optional[float] = None
    """Per-rollout outcome reward."""

    segment_rewards: Optional[List[float]] = None
    """Per-segment rewards. Populated only when the reward function returns
    segment-level scores; ``None`` for trajectories where reward is per-rollout."""

    metrics: Dict[str, Any] = Field(default_factory=dict)
    """Extra fields emitted alongside the main reward by the reward function
    (e.g. ``f1``, ``em``, ``format``, or non-scalar fields like ``output``).

    Numeric values are logged as ``reward_extra/<key>/{mean,max,min}`` by the
    trainer; non-numeric values pass through and are available for inspection
    on the trajectory but not aggregated."""

    # Outcome (set by the chain runtime)
    finish_reason: Optional[str] = None
    """One of ``"terminal"``, ``"max_turns"``, ``"max_model_len"``,
    ``"no_tool_calls"``, etc."""

    rollout_time_sec: Optional[float] = None

    # Identifiers (set at chain construction)
    chain_id: Optional[str] = None
    group_id: Optional[str] = None
    chain_idx: Optional[int] = None
    group_idx: Optional[int] = None

    # Free-form bags
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Task-level fields preserved from the dataset row (e.g. ``answer``,
    ``task_id``, ``fen``)."""

    runtime_info: Dict[str, Any] = Field(default_factory=dict)
    """Escape hatch for runtime-produced state that doesn't fit the typed
    fields above. Use sparingly; prefer first-class fields when a value is
    stable."""

    # ---- Convenience accessors ----

    @property
    def is_segmented(self) -> bool:
        """True if this trajectory has more than one segment (e.g. context
        folding was applied during the rollout)."""
        return len(self.segments) > 1

    @property
    def num_segments(self) -> int:
        return len(self.segments)


class RunResult(BaseModel):
    """Result of a single ``agent.run(...)`` call.

    Holds the per-rollout trajectories plus convenience views over rewards
    and extra metrics. ``RunResult`` is iterable and supports ``len()`` so
    consumers can write ``for t in result: ...`` and ``result[i]``.
    """

    trajectories: List[Trajectory]

    # ---- Convenience accessors ----

    @property
    def rewards(self) -> List[Optional[float]]:
        """Per-rollout main reward, in trajectory order."""
        return [t.reward for t in self.trajectories]

    @property
    def reward_extras(self) -> Dict[str, List[Any]]:
        """Extra scalar metrics aggregated across trajectories.

        For each key that appears in *any* trajectory's ``metrics``, returns
        a list of length ``len(trajectories)`` with that key's value per
        trajectory (``None`` where the trajectory's metrics didn't include
        the key). Keys are returned in sorted order.
        """
        keys: set[str] = set()
        for t in self.trajectories:
            keys.update(t.metrics.keys())
        return {
            k: [t.metrics.get(k) for t in self.trajectories]
            for k in sorted(keys)
        }

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self):
        return iter(self.trajectories)

    def __getitem__(self, i: int) -> Trajectory:
        return self.trajectories[i]
