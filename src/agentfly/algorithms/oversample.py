"""Oversample-and-filter group selection for GRPO (the team's dynamic-sampling method).

With ``data.gen_batch_size`` set to a multiple of ``data.train_batch_size``, the trainer
generates more prompt groups per step than it trains on. :func:`apply_oversample_filter`
then keeps exactly ``train_batch_size`` groups, **preferring groups whose episodes got
different rewards** (nonzero variance — the only groups that yield a GRPO gradient) and
padding with zero-variance groups only when too few are found. So the training batch size
stays constant every step, unlike the pure zero-variance mask (``grpo_episode``'s
``filter_zero_variance_groups``), whose effective batch shrinks when many groups are flat.

This mirrors the team's ``usable_and_nonzero_std`` filter: zero-variance groups are
dropped first and, when a per-row ``infra_failed`` flag is available, rows that hit an
infrastructure failure are ignored when judging a group's variance (an infra failure is not
a real reward, so a genuine solve paired with a spurious infra zero is not read as signal).

Only one hook lives in the trainer (``apply_oversample_filter``); all the selection logic is
here. Enabled by ``algorithm.filter_groups.enable`` with optional ``algorithm.filter_groups.metric``
(a key in the reward-extra dict, e.g. ``acc``; falls back to the sequence reward).
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("agentfly")


def select_group_indices(
    seq_reward: Sequence[float],
    uids: Sequence[Any],
    train_batch_size: int,
    infra_failed: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, int, int]:
    """``(row_indices, n_total_groups, n_usable_groups)``: rows of the ``train_batch_size`` chosen
    groups (variance groups first), plus how many groups were generated and how many of those had
    variance. The counts let a caller log the usable rate and tune the oversample factor.

    ``seq_reward`` is one scalar per row; ``uids`` (same length) is each row's group id (shared
    across a prompt's rollouts). A group has variance when its usable rewards are not all equal.
    ``infra_failed`` (truthy per row) marks rows to ignore when judging variance. Rows are
    returned in their original order so tensors stay aligned."""
    r = np.asarray(seq_reward, dtype=float)
    uids = list(uids)
    infra = None if infra_failed is None else [bool(x) for x in infra_failed]

    rows_by_uid: Dict[Any, List[int]] = {}
    order: List[Any] = []
    for i, u in enumerate(uids):
        if u not in rows_by_uid:
            rows_by_uid[u] = []
            order.append(u)
        rows_by_uid[u].append(i)

    variance: List[Any] = []
    flat: List[Any] = []
    for u in order:
        rows = rows_by_uid[u]
        usable = [i for i in rows if infra is None or not infra[i]]
        vals = r[usable] if len(usable) >= 2 else r[rows]
        has_var = len(usable) >= 2 and float(np.std(vals)) > 0.0
        (variance if has_var else flat).append(u)

    chosen = set((variance + flat)[: max(0, int(train_batch_size))])
    idxs = np.asarray([i for i, u in enumerate(uids) if u in chosen], dtype=np.int64)
    return idxs, len(order), len(variance)


def _seq_reward(reward_tensor, reward_extra_infos_dict: Dict[str, List], metric: Optional[str]):
    """Per-row scalar used to judge variance: the reward-extra ``metric`` column when present,
    else the summed token-level reward."""
    if metric and reward_extra_infos_dict and metric in reward_extra_infos_dict:
        return [float(x) for x in reward_extra_infos_dict[metric]]
    r = reward_tensor
    if hasattr(r, "dim"):
        if r.dim() > 1:
            r = r.sum(dim=-1)
        return [float(x) for x in r.detach().cpu().tolist()]
    r = np.asarray(r, dtype=float)
    return (r.sum(axis=-1) if r.ndim > 1 else r).tolist()


def apply_oversample_filter(config, batch, reward_tensor, reward_extra_infos_dict):
    """Downselect an oversampled ``batch`` to a constant ``train_batch_size`` groups.

    No-op unless ``algorithm.filter_groups.enable`` is set. Returns the (possibly) filtered
    ``(batch, reward_tensor, reward_extra_infos_dict)``; the batch's ``global_token_num`` is
    recomputed so later stages see the reduced batch. Called once in the trainer, right after
    the reward is extracted and before the expensive log-prob / advantage work."""
    fg = config.algorithm.get("filter_groups", None)
    if not fg or not fg.get("enable", False):
        return batch, reward_tensor, reward_extra_infos_dict
    if "uid" not in batch.non_tensor_batch:
        return batch, reward_tensor, reward_extra_infos_dict

    train_bs = int(config.data.train_batch_size)
    metric = fg.get("metric", None)
    seq_reward = _seq_reward(reward_tensor, reward_extra_infos_dict, metric)
    infra = (reward_extra_infos_dict or {}).get("infra_failed")
    idxs, n_total, n_usable = select_group_indices(
        seq_reward, batch.non_tensor_batch["uid"], train_bs, infra_failed=infra
    )
    n_kept = min(n_total, train_bs)
    n_padded = max(0, n_kept - n_usable)
    logger.info(
        f"[oversample] generated {n_total} groups, {n_usable} usable (nonzero-variance), "
        f"kept {n_kept} (padded {n_padded} zero-variance to fill the batch)"
    )

    if len(idxs) == 0 or len(idxs) == len(seq_reward):
        return batch, reward_tensor, reward_extra_infos_dict  # nothing to drop

    # The trainer splits rows evenly over the GPUs, and the conversion made the row count a multiple of
    # ppo_mini_batch_size for that. An agent with subagents has several rows per episode, so the kept rows are no
    # longer such a multiple: pad the same way, by repeating the last row with its action mask set to zero.
    multiple = int(config.actor_rollout_ref.actor.get("ppo_mini_batch_size", 0) or 0)
    pad = (-len(idxs)) % multiple if multiple > 1 else 0
    if pad:
        idxs = np.concatenate([idxs, np.full(pad, idxs[-1], dtype=idxs.dtype)])

    batch = batch.select_idxs(idxs)
    reward_tensor = reward_tensor[idxs]
    if pad and "action_mask" in batch.batch and not batch.batch["action_mask"].is_nested:
        batch.batch["action_mask"][-pad:] = 0  # as the conversion's padding: a copy of the last row, out of the loss
    reward_extra_infos_dict = {k: [v[i] for i in idxs] for k, v in (reward_extra_infos_dict or {}).items()}
    if "attention_mask" in batch.batch:
        import torch

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
    return batch, reward_tensor, reward_extra_infos_dict
