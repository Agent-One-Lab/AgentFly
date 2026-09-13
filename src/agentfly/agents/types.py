"""Typed data classes for agent rollouts.

The framework moves data between agent, chain, tools, rewards, and trainer
through these classes. User-authored code (`@tool`, `@reward` functions) does
not need to construct them — the framework normalizes user returns at the
boundary via the `from_raw` factories on `tools.types.ToolResult` and
`rewards.types.RewardResult`.

Three classes are exposed here:

- :class:`Segment` — one model-context view recorded during rollout.
- :class:`Trajectory` — one rollout's worth of data (conversation segments
  plus its reward signal, metrics, identifiers, and free-form metadata).
- :class:`RunResult` — what `agent.run(...)` returns: a list of trajectories
  with convenience accessors.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Segment(BaseModel):
    """One model-context view recorded during rollout.

    ``messages`` contains context and generated responses in their original
    message format, including any sampled ``token_ids``, tool calls, and
    multimodal content. Consumers read ``segment.messages`` explicitly.

    Segments may overlap: a later view can repeat earlier messages as context.
    A segment is neither an environment step nor a disjoint transcript slice.
    Its position alone does not specify training targets or reward assignment.
    Empty or context-only views are valid rollout data. Training conversion
    selects views with policy-token targets without removing any from the result.
    """

    messages: List[Dict[str, Any]]


class Trajectory(BaseModel):
    """One task attempt, containing its ordered model-context views.

    ``segments`` is the canonical store of conversation data. A chain normally
    has one segment; context folding preserves the pre-fold views and the final
    view as separate segments. A step rollout has one prompt-plus-response
    segment per generation. In either case, consumers read each segment's
    ``messages`` rather than concatenate segments into an episode transcript.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Conversation data
    segments: List[Segment]

    # RL signals
    reward: Optional[float] = None
    """Per-rollout outcome reward."""

    segment_rewards: Optional[List[float]] = None
    """Per-segment rewards. Populated only when the reward function returns
    segment-level scores; ``None`` for trajectories where reward is per-rollout."""

    metrics: Dict[str, Any] = Field(default_factory=dict)
    """Reward-function extras plus aggregated numeric tool metrics.

    Reward extras include fields like ``f1``, ``em``, ``format``, or non-scalar
    ``output``. Tool aggregates use ``tool/<name>/<key>`` and take precedence on
    a collision with a reward extra in that namespace, for both rollout strategies.

    Numeric values are logged as ``reward_extra/<key>/{mean,max,min}`` by the
    trainer; non-numeric values pass through and are available for inspection
    on the trajectory but not aggregated."""

    # Outcome (set by the rollout runtime)
    finish_reason: Optional[str] = None
    """Built-in rollouts use the shared ``FinishReason`` vocabulary: ``terminal``
    (model-requested stop), ``tool_control_end``, ``invalid_end``, ``tool_error_end``,
    ``tool_end``, ``max_turns``, ``max_model_len``, or ``no_tool_calls``.
    Custom rollouts may supply their own strings."""

    rollout_time_sec: Optional[float] = None

    num_turns: Optional[int] = None
    """Completed model generations, excluding repeated prompt context.

    ``None`` means unavailable (e.g. an older/custom result); zero means measured
    with no completed generations. Terminal responses without tools count.
    """

    tool_call_counts: Optional[Dict[str, int]] = None
    """Recorded tool-call attempts grouped by ``ToolResult.name``.

    Includes invalid/error results, not just successful executions. Proposed calls
    that never reach execution are excluded. ``None`` means unavailable; an empty
    dict means measured with no calls. The total is the sum of these counts.
    """

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

    steps: List[Any] = Field(default_factory=list, exclude=True)
    """Internal compatibility channel for rollout-specific training signals.

    Chain rollouts store tool-result steps; step rollouts store generation steps,
    including generations without a tool result. These are not a uniform public
    step protocol. Conversation consumers must use ``segments`` instead.

    Typed ``List[Any]`` and ``exclude``d from serialization on purpose: a ``Step`` holds
    live tree links (``parent``/``children``) and a ``Messages`` object, so it is an
    in-memory harvest channel read via attribute access (``traj.steps[i].tool_result``),
    never through ``model_dump()``/JSON."""

    # ---- Convenience accessors ----

    @property
    def is_segmented(self) -> bool:
        """Whether there are multiple context views, not whether folding occurred."""
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

    rollout: Optional[str] = None
    """Identifier of the rollout that produced this result, not a runtime object.

    Built-in rollouts set ``"chain"`` or ``"step"``. Conversion uses this
    identifier to select its stateless exporter; missing or unsupported identifiers
    are errors at that boundary. ``None`` permits inference-only/manual results.
    The identifier is preserved in serialization; runtime ``Trajectory.steps``
    remain excluded, so serialization alone is not a complete training snapshot.
    """

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
