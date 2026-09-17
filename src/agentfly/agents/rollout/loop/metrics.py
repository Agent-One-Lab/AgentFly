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


DEFAULT_SUCCESS_THRESHOLD = 1.0


def group_rewards(trajectories: Iterable[Trajectory]) -> Dict[str, List[float]]:
    """Group per-rollout rewards by task, keyed by ``group_id``.

    Both strategies stamp every rollout of one task with a shared ``group_id``, so this
    reconstructs the sample set per task. Rollouts without a reward (no reward function,
    or an ungradable attempt) are skipped; a rollout with no ``group_id`` forms its own
    single-sample group rather than being merged with other ungrouped rollouts.
    """
    grouped: Dict[str, List[float]] = defaultdict(list)
    for index, trajectory in enumerate(trajectories):
        if trajectory.reward is None:
            continue
        key = trajectory.group_id if trajectory.group_id is not None else f"__ungrouped_{index}"
        grouped[key].append(float(trajectory.reward))
    return dict(grouped)


def pass_rate_distribution(
    rewards_by_group: Dict[str, List[float]],
    *,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> Dict[str, Any]:
    """Summarize the per-task pass-rate distribution of one rollout batch.

    With several rollouts per task, each task has a *pass rate* — the fraction of its
    samples reaching ``success_threshold``. The shape of that distribution is what says
    whether a task pool is trainable: a U-shape (mass at 0 and 1) means the pool is
    mostly tasks the policy always fails and tasks it always solves, and a group whose
    samples all score alike gives GRPO no advantage signal at all, so those steps
    contribute nothing to the gradient regardless of how good the mean reward looks.

    Returns the per-group pass rates plus summary fractions. ``frac_zero_advantage``
    and ``reward_std_mean`` read the raw rewards and so stay meaningful for continuous
    rewards, where the threshold-based fractions can be misleading.
    """
    pass_rates: List[float] = []
    reward_stds: List[float] = []
    zero_advantage = 0
    for rewards in rewards_by_group.values():
        passes = sum(1 for reward in rewards if reward >= success_threshold)
        pass_rates.append(passes / len(rewards))
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        reward_stds.append(variance ** 0.5)
        if max(rewards) == min(rewards):
            zero_advantage += 1

    num_groups = len(pass_rates)
    if not num_groups:
        return {"pass_rates": [], "num_groups": 0}
    return {
        "pass_rates": pass_rates,
        "num_groups": num_groups,
        "samples_per_task": sum(len(r) for r in rewards_by_group.values()) / num_groups,
        "pass_rate_mean": sum(pass_rates) / num_groups,
        "frac_all_fail": sum(1 for rate in pass_rates if rate == 0.0) / num_groups,
        "frac_all_pass": sum(1 for rate in pass_rates if rate == 1.0) / num_groups,
        "frac_mixed": sum(1 for rate in pass_rates if 0.0 < rate < 1.0) / num_groups,
        "frac_zero_advantage": zero_advantage / num_groups,
        "reward_std_mean": sum(reward_stds) / num_groups,
    }


class RolloutMetrics:
    """Emit each report from its supplied data without accumulating chain state."""

    def __init__(self, success_threshold: float = DEFAULT_SUCCESS_THRESHOLD) -> None:
        """``success_threshold`` is the reward at which a single rollout counts as a
        pass for the per-task pass-rate distribution."""
        self.success_threshold = success_threshold

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
                    x=global_step,
                    x_name="agent/rollout/step",
                )
                emit(evt)

        self._record_pass_rate_distribution(
            global_step=global_step, trajectories=trajectories
        )

    def _record_pass_rate_distribution(
        self,
        *,
        global_step: int,
        trajectories: List[Trajectory],
    ) -> None:
        """Report how this batch's per-task pass rates are distributed.

        Emitted only when some task actually has several rollouts: with one sample per
        task (validation, or ``num_chains=1``) every pass rate is 0 or 1 by construction
        and the distribution would say nothing about task difficulty. The raw per-task
        pass rates go out as a histogram so the shape — U-shaped means a pool of
        always-failed and always-solved tasks, i.e. little to learn from — is visible
        per step, alongside the scalar fractions that plot as curves over training.
        """
        rewards_by_group = group_rewards(trajectories)
        if max((len(r) for r in rewards_by_group.values()), default=0) < 2:
            return
        distribution = pass_rate_distribution(
            rewards_by_group, success_threshold=self.success_threshold
        )
        pass_rates = distribution.pop("pass_rates")

        logger.info(
            "Pass-rate distribution over %d tasks (%.1f samples each): "
            "mean=%.3f all-fail=%.3f mixed=%.3f all-pass=%.3f zero-advantage=%.3f",
            distribution["num_groups"],
            distribution["samples_per_task"],
            distribution["pass_rate_mean"],
            distribution["frac_all_fail"],
            distribution["frac_mixed"],
            distribution["frac_all_pass"],
            distribution["frac_zero_advantage"],
        )

        for key, value in distribution.items():
            emit(
                MetricEvent(
                    kind="scalar",
                    name=f"agent/rollout/group/{key}",
                    value=value,
                    x=global_step,
                    x_name="agent/rollout/step",
                )
            )
        emit(
            MetricEvent(
                kind="hist",
                name="agent/rollout/group/pass_rate_hist",
                value=pass_rates,
                x=global_step,
                x_name="agent/rollout/step",
            )
        )
