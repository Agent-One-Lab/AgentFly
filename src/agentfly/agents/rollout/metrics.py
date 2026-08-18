"""Rollout metrics — a consumer of the rollout's progress, separated from the loop.

``RolloutMetrics`` owns the metric aggregation/emission that used to live inside
``ChainRollout`` as ``monitor_chain`` / ``monitor_step``. The loop feeds it:

- :meth:`record_chain` once per chain as it ends (the ``ChainEnded`` point), and
- :meth:`record_step` once the whole rollout is drained.

All the rollout state it needs is passed in explicitly rather than reached through
``self`` of the rollout, so the metrics logic is testable in isolation (feed synthetic
trajectories) and the loop class no longer carries ~210 lines of reporting.
"""

import json
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List

from ...utils.monitor import MetricEvent, emit, serialize_for_json

logger = logging.getLogger(__name__)


class RolloutMetrics:
    def __init__(self) -> None:
        # Accumulated across chains within a step; read in record_step.
        self.monitor_info: Dict[str, list] = defaultdict(list)

    def record_chain(
        self,
        *,
        global_step: int,
        finished_chains_count: int,
        trajectory: Any,
        info: Any,
    ) -> None:
        """Per-chain reporting (was ``ChainRollout.monitor_chain``).

        Logs the full trajectory + info to the local JSONL sink only — emitting these to
        wandb every chain would cost too much bandwidth.
        """
        self.monitor_info["agent/chains"].append(finished_chains_count)

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
        trajectories: List[Any],
        chain_rollout_seconds: Dict[str, float],
        current_nodes: Dict[str, Any],
        chains: Dict[str, Any],
    ) -> None:
        """End-of-rollout aggregation (was ``ChainRollout.monitor_step``)."""
        if chain_rollout_seconds:
            vals = list(chain_rollout_seconds.values())
            min_t = min(vals)
            max_t = max(vals)
            avg_t = sum(vals) / len(vals)

            logger.info(
                "Chain rollout time stats (s): count=%d min=%.3f max=%.3f avg=%.3f",
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

        avg_turns = 0
        avg_tool_calls = 0
        avg_segments = 0
        tool_calls_by_name: Dict[str, int] = defaultdict(int)

        for trajectory in trajectories:
            for segment in trajectory.segments:
                avg_segments += 1
                for msg in segment:
                    if msg["role"] == "assistant":
                        avg_turns += 1
                        if "tool_calls" in msg:
                            for tool_call in msg["tool_calls"]:
                                tool_call_name = tool_call["function"]["name"]
                                if tool_call_name in ["summarize"]:
                                    tool_calls_by_name["summarize"] += 1

                    if msg["role"] == "tool":
                        avg_tool_calls += 1
                        tool_call_name = msg["tool_name"]
                        tool_calls_by_name[tool_call_name] += 1

        avg_turns /= len(trajectories)
        avg_tool_calls /= len(trajectories)
        avg_segments /= len(trajectories)

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

        evt = MetricEvent(
            kind="scalar",
            name="agent/rollout/avg_turns",
            value=avg_turns,
            x=global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)

        evt = MetricEvent(
            kind="scalar",
            name="agent/rollout/avg_tool_calls",
            value=avg_tool_calls,
            x=global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)

        for tool_name, tool_call_count in tool_calls_by_name.items():
            evt = MetricEvent(
                kind="scalar",
                name=f"agent/rollout/tool_calls/{tool_name}",
                value=tool_call_count / len(trajectories),
                x=global_step,
                x_name="agent/rollout/step",
            )
            emit(evt)

        evt = MetricEvent(
            kind="scalar",
            name="agent/rollout/step",
            value=global_step,
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

        if chain_rollout_seconds:
            slowest_chain_id = max(
                chain_rollout_seconds, key=chain_rollout_seconds.get
            )
            if slowest_chain_id in current_nodes and slowest_chain_id in chains:
                slowest_message = {
                    "chain_id": slowest_chain_id,
                    "rollout_time_sec": chain_rollout_seconds[slowest_chain_id],
                    "messages": current_nodes[slowest_chain_id].messages.messages,
                    **chains[slowest_chain_id].info,
                }
                slowest_message_json = json.dumps(
                    serialize_for_json(slowest_message), indent=2
                )
                evt = MetricEvent(
                    kind="text",
                    name="agent/rollout/slowest_message",
                    value=slowest_message_json,
                    x=global_step,
                    x_name="agent/rollout/step",
                )
                emit(evt)

        for k, v in self.monitor_info.items():
            if k != "agent/chains":  # We don't log number of chains
                evt = MetricEvent(
                    kind="list",
                    name=k,
                    value=v,
                    x=self.monitor_info["agent/chains"],
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
