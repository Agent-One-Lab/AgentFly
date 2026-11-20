"""Per-step check that the trainer scores what the rollout engine sampled.

``rollout_parity_metrics`` is an ``algorithm.metrics`` function (``agentfly.algorithms.rollout_parity:
rollout_parity_metrics``). It needs the engine's log-probabilities in the batch (``engine_log_probs``, written
by the rollout conversion when ``actor_rollout_ref.rollout.calculate_log_probs=True``) and compares them with
the trainer's ``old_log_probs`` on the sampled tokens. With routing replay it also reports how much of the batch
carries recorded expert choices.

Layouts: ``engine_log_probs[i, p]`` is the engine's log-probability of the token at position ``p`` (NaN where it
reported none); ``old_log_probs[i, t]`` is the trainer's for the token at ``t + 1``, and ``action_mask[i, t]``
(as the trainer holds it when metrics run) says whether that token was sampled.

A dense model shows a mean gap around 0.01 (bf16 rounding). A routed model without replay is several times
higher; a gap of whole nats means the two are not scoring the same sequence.
"""

from typing import Any, Dict

import torch

from .registry import register_metrics  # noqa: F401  (kept importable next to the other metric modules)

PAD = 255


def rollout_parity_metrics(batch: Any, algorithm_config: Any = None) -> Dict[str, float]:
    data = batch.batch
    metrics: Dict[str, float] = {}
    record = (getattr(batch, "meta_info", None) or {}).get("rollout_record")
    if record:
        metrics["rollout_parity/rows_left_out"] = float(record.get("mismatched", 0))
    if "routed_experts" in data.keys():
        routed = data["routed_experts"]
        positions = data["attention_mask"].bool()
        recorded = (routed != PAD).flatten(2).any(-1) & positions
        metrics["rollout_parity/positions_with_routes"] = float(recorded.sum() / positions.sum().clamp(min=1))
    if "engine_log_probs" not in data.keys() or "old_log_probs" not in data.keys():
        return metrics
    engine = data["engine_log_probs"][:, 1:].float()
    trainer = data["old_log_probs"][:, :-1].float()
    valid = torch.isfinite(engine)
    if "action_mask" in data.keys():
        # by the time metrics run, the trainer has shifted action_mask to the same layout as old_log_probs
        trained = data["action_mask"][:, :-1].bool()
        metrics["rollout_parity/sampled_tokens_scored_by_engine"] = float(
            (valid & trained).sum() / trained.sum().clamp(min=1)
        )
        valid = valid & trained
    if not valid.any():
        return metrics
    gap = (trainer - engine)[valid]
    absolute = gap.abs()
    metrics.update(
        {
            "rollout_parity/logprob_gap_mean": float(absolute.mean()),
            "rollout_parity/logprob_gap_p99": float(torch.quantile(absolute, 0.99)) if absolute.numel() < 2**24
            else float(absolute.float().kthvalue(int(absolute.numel() * 0.99)).values),
            "rollout_parity/logprob_gap_max": float(absolute.max()),
            "rollout_parity/logprob_gap_over_0.05": float((absolute > 0.05).float().mean()),
            "rollout_parity/logprob_gap_signed_mean": float(gap.mean()),
        }
    )
    return metrics
