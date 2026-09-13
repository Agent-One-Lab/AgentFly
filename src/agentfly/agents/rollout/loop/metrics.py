"""Rollout metrics — a consumer of the rollout's progress, separated from the loop.

``RolloutMetrics`` owns the metric aggregation/emission that used to live inside
``ChainRollout`` as ``monitor_chain`` / ``monitor_step``. The loop feeds it:

- :meth:`record_chain` once per chain as it ends (the ``ChainEnded`` point), and
- :meth:`record_step` once the whole rollout is drained.

End-of-rollout reporting consumes typed trajectories, not chain runtime dictionaries.
Both strategies supply the same data for timing summaries and slowest-trajectory
logging; the reporting logic can also be tested directly with synthetic trajectories.
"""

import json
import logging
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

from ....tools.types import ToolResult
from ....utils.monitor import MetricEvent, emit, serialize_for_json
from ...types import Trajectory

logger = logging.getLogger(__name__)


def count_tool_calls(results: Iterable[Optional[ToolResult]]) -> Dict[str, int]:
    """Count recorded attempts by result name, including invalid/error results."""
    return dict(Counter(result.name for result in results if result is not None))


def aggregate_tool_metrics(results: Iterable[Optional[ToolResult]]) -> Dict[str, float]:
    """Average supplied numeric tool metrics under ``tool/<name>/<key>``.

    Consume ToolResults, not runtime Steps, so each rollout can provide its own
    records without assuming a shared step granularity. Missing results and
    non-numeric values are ignored, not treated as zero. Booleans count as 0/1.
    The first-class ``step_reward`` is also included when present, following the
    existing chain aggregation rule. Inputs are never modified.
    """
    acc: Dict[str, List[Any]] = {}
    for result in results:
        if result is None:
            continue
        name = result.name or "tool"
        for key, value in (result.metrics or {}).items():
            acc.setdefault(f"tool/{name}/{key}", []).append(value)
        if result.step_reward is not None:
            acc.setdefault(f"tool/{name}/step_reward", []).append(result.step_reward)

    averages: Dict[str, float] = {}
    for key, values in acc.items():
        nums = [float(value) for value in values if isinstance(value, (int, float, bool))]
        if nums:
            averages[key] = sum(nums) / len(nums)
    return averages


class RolloutMetrics:
    """Emit each report from its supplied data without accumulating chain state."""

    def record_chain(
        self,
        *,
        global_step: int,
        trajectory: Any,
        info: Any,
    ) -> None:
        """Per-chain reporting (was ``ChainRollout.monitor_chain``).

        Logs the full trajectory + info to the local JSONL sink only — emitting these to
        wandb every chain would cost too much bandwidth.
        """
        evt = MetricEvent(
            sinks=["jsonl"],
            kind="text",
            name="agent/rollout/trajectory",
            value=json.dumps(serialize_for_json(trajectory), indent=2),
            x=global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)

        evt = MetricEvent(
            sinks=["jsonl"],
            kind="text",
            name="agent/rollout/info",
            value=json.dumps(serialize_for_json(info), indent=2),
            x=global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)

    def record_step(
        self,
        *,
        global_step: int,
        trajectories: List[Trajectory],
    ) -> None:
        """Report a completed batch without retaining or mutating its trajectories.

        Empty batches emit no summaries. Missing durations are omitted from timing
        statistics; zero is valid. Duration ties select the first trajectory in the
        supplied order for detailed logging. Execution-count averages omit missing
        fields independently; measured zeros are included. Segments are context
        views, not execution records, and are used only for the segment count.
        """
        if not trajectories:
            return

        timed_trajectories = [t for t in trajectories if t.rollout_time_sec is not None]
        if timed_trajectories:
            vals = [t.rollout_time_sec for t in timed_trajectories]
            min_t = min(vals)
            max_t = max(vals)
            avg_t = sum(vals) / len(vals)

            logger.info(
                "Rollout time stats (s): count=%d min=%.3f max=%.3f avg=%.3f",
                len(vals),
                min_t,
                max_t,
                avg_t,
            )

            for name, value in (
                ("agent/rollout/time/min", min_t),
                ("agent/rollout/time/max", max_t),
                ("agent/rollout/time/avg", avg_t),
            ):
                emit(
                    MetricEvent(
                        kind="scalar",
                        name=name,
                        value=value,
                        x=global_step,
                        x_name="agent/rollout/step",
                    )
                )

        turn_counts = [t.num_turns for t in trajectories if t.num_turns is not None]
        tool_counts = [t.tool_call_counts for t in trajectories if t.tool_call_counts is not None]
        avg_segments = sum(t.num_segments for t in trajectories) / len(trajectories)

        ent = MetricEvent(
            kind="scalar",
            name="agent/rollout/avg_segments",
            value=avg_segments,
            x=global_step,
            x_name="agent/rollout/step",
        )
        emit(ent)

        ent = MetricEvent(
            kind="scalar",
            name="agent/rollout/step",
            value=global_step,
            x=global_step,
            x_name="agent/rollout/step",
        )
        emit(ent)

        if turn_counts:
            evt = MetricEvent(
                kind="scalar",
                name="agent/rollout/avg_turns",
                value=sum(turn_counts) / len(turn_counts),
                x=global_step,
                x_name="agent/rollout/step",
            )
            emit(evt)

        if tool_counts:
            tool_calls_by_name: Counter[str] = Counter()
            for counts in tool_counts:
                tool_calls_by_name.update(counts)
            evt = MetricEvent(
                kind="scalar",
                name="agent/rollout/avg_tool_calls",
                value=sum(tool_calls_by_name.values()) / len(tool_counts),
                x=global_step,
                x_name="agent/rollout/step",
            )
            emit(evt)

            for tool_name, tool_call_count in tool_calls_by_name.items():
                evt = MetricEvent(
                    kind="scalar",
                    name=f"agent/rollout/tool_calls/{tool_name}",
                    value=tool_call_count / len(tool_counts),
                    x=global_step,
                    x_name="agent/rollout/step",
                )
                emit(evt)

        sample_trajectory_json = json.dumps(
            serialize_for_json(random.choice(trajectories)), indent=2
        )
        evt = MetricEvent(
            kind="text",
            name="agent/rollout/sample_trajectory",
            value=sample_trajectory_json,
            x=global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)

        if timed_trajectories:
            slowest_trajectory = max(
                timed_trajectories, key=lambda t: t.rollout_time_sec
            )
            # Keep the event name; its payload now matches sample_trajectory and
            # preserves every context segment. Runtime-only steps stay excluded.
            evt = MetricEvent(
                kind="text",
                name="agent/rollout/slowest_message",
                value=json.dumps(serialize_for_json(slowest_trajectory), indent=2),
                x=global_step,
                x_name="agent/rollout/step",
            )
            emit(evt)

        # Compute reward + extras from the typed Trajectory objects. Chains without a
        # reward function produce reward=None; filter those out before averaging.
        reward_values = [t.reward for t in trajectories if t.reward is not None]
        other_values: Dict[str, List[Any]] = defaultdict(list)
        for t in trajectories:
            for k, v in t.metrics.items():
                other_values[k].append(v)

        if reward_values:
            avg_reward = sum(reward_values) / len(reward_values)
            evt = MetricEvent(
                kind="scalar",
                name="agent/rollout/reward",
                value=avg_reward,
                x=global_step,
                x_name="agent/rollout/step",
            )
            emit(evt)
        for key, value in other_values.items():
            if value and isinstance(value[0], (float, int)) and not isinstance(value[0], bool):
                avg_value = sum(value) / len(value)
                evt = MetricEvent(
                    kind="scalar",
                    name=f"agent/rollout/{key}",
                    value=avg_value,
                )
                emit(evt)
