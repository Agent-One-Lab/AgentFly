"""Shared building blocks for group-relative agent advantage estimators.

The grouping and normalisation functions (``to_hashable``, ``are_similar``,
``episode_norm_reward``, ``build_step_group``, ``step_norm_reward``) and the discounted
step returns are verl-agent's (``gigpo/core_gigpo.py``), kept verbatim so every layout
normalises exactly like verl-agent: sample standard deviation, an episode group with a
single row gets mean 0 / std 1, a step group with a single row gets its own mean.
``tests/unit/algorithms/test_gigpo.py`` checks bit-identity against verl-agent's code.

The remaining helpers adapt AgentFly batches to those functions: batch-field access,
turn spans inside a multi-turn row, and the terminal response token.
"""

import uuid
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


# ------------------------------------------------------------------------------------ #
# verl-agent verbatim (gigpo/core_gigpo.py)                                             #
# ------------------------------------------------------------------------------------ #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO.")
    return SequenceMatcher(None, a, b).ratio() >= threshold


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True):
    """Episode-level advantage (mean/std over a uid group's rows). verl-agent verbatim."""
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False,
                     similarity_thresh: float = 0.95, summarize: bool = False):
    """Cluster rows with identical ``anchor_obs`` within a uid group. verl-agent verbatim."""
    if enable_similarity:
        assert 0.0 < similarity_thresh < 1.0, "similarity_thresh should be in (0, 1)"

    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    unique_indices = np.unique(index)
    group_size: List[int] = []
    for idx in unique_indices:
        if not enable_similarity:
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])
            for obs, original_indices in clusters.items():
                uid = str(uuid.uuid4())
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]
            clusters: List[Dict[str, Any]] = []
            for obs, loc in zip(obs_group, locs):
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

    if None in step_group_uids or np.any(step_group_uids == None):  # noqa: E711
        missing = np.where(step_group_uids == None)[0]  # noqa: E711
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at: {missing}")

    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                     response_mask: torch.Tensor,
                     index: np.array,
                     epsilon: float = 1e-6,
                     remove_std: bool = True):
    """Step-level advantage (mean/std over an anchor step-group). verl-agent verbatim."""
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    return step_advantages


def compute_step_discounted_returns(step_rewards: np.ndarray, traj_uids: np.ndarray,
                                    gamma: float) -> np.ndarray:
    """Per-``traj_uid`` backward discounted return ``G_t = r_t + gamma * G_{t+1}``.

    verl-agent's ``compute_step_discounted_returns`` on raw arrays (rows of one trajectory
    appear in step order).
    """
    rewards = step_rewards.astype(np.float32)
    returns_by_traj = {}
    for uid in np.unique(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        traj_rewards = rewards[traj_indices]
        traj_returns = np.zeros_like(traj_rewards)
        running = 0.0
        for t in reversed(range(len(traj_rewards))):
            running = traj_rewards[t] + gamma * running
            traj_returns[t] = running
        returns_by_traj[uid] = traj_returns

    all_returns = np.zeros_like(rewards)
    for uid in np.unique(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        all_returns[traj_indices] = returns_by_traj[uid]
    return all_returns


# ------------------------------------------------------------------------------------ #
# AgentFly adapters                                                                     #
# ------------------------------------------------------------------------------------ #
NORM_MODES = {"mean_std_norm": False, "mean_norm": True}  # mode -> verl-agent's ``remove_std``


def remove_std_for(mode: str) -> bool:
    """verl-agent's ``remove_std`` flag for a normalisation mode name."""
    if mode not in NORM_MODES:
        raise ValueError(f"Unknown normalisation mode {mode!r}; expected one of {sorted(NORM_MODES)}.")
    return NORM_MODES[mode]


def config_value(config: Any, name: str, default: Any) -> Any:
    """``config.<name>`` for a dataclass or DictConfig algorithm config, else ``default``."""
    if config is None:
        return default
    if hasattr(config, "get"):
        value = config.get(name, default)
    else:
        value = getattr(config, name, default)
    return default if value is None else value


def batch_field(data: Any, name: str, estimator: str, layout: str, optional: bool = False):
    """A tensor or non-tensor batch field, with a clear error if the rollout did not emit it."""
    if name in data.batch.keys():
        return data.batch[name]
    if name in data.non_tensor_batch:
        return data.non_tensor_batch[name]
    if optional:
        return None
    raise KeyError(
        f"{estimator} on a {layout!r} batch needs the field {name!r}, which the rollout's "
        f"training conversion did not provide."
    )


def compute_turn_ids(response_mask: torch.Tensor) -> torch.Tensor:
    """Map each token to its turn index, derived from the response mask.

    A turn is one contiguous run of policy (assistant) tokens; a ``0 -> 1`` transition starts
    a new turn. Returns ``[B, L]`` int64: turn index ``0..n-1`` on that turn's tokens, ``-1``
    on non-policy tokens.
    """
    am = (response_mask > 0).long()
    prev = torch.cat([torch.zeros_like(am[:, :1]), am[:, :-1]], dim=1)
    starts = (am == 1) & (prev == 0)
    turn_idx = torch.cumsum(starts.long(), dim=1) - 1
    return torch.where(am == 1, turn_idx, torch.full_like(turn_idx, -1))


def terminal_token_mask(response_mask: torch.Tensor) -> torch.Tensor:
    """One-hot ``[B, L]`` marking the last response token of each row (0 if a row has none)."""
    am = response_mask > 0
    L = am.shape[-1]
    positions = torch.arange(L, device=am.device).unsqueeze(0)
    last = torch.where(am, positions, torch.full_like(positions, -1)).max(dim=1).values
    out = torch.zeros_like(response_mask)
    valid = last >= 0
    out[valid, last[valid]] = 1
    return out


def object_array(values: Sequence[Any]) -> np.ndarray:
    """A 1-D ``dtype=object`` array holding ``values`` as-is (lists are not expanded)."""
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


def print_group_diagnostics(tag: str, step_group_uids: np.ndarray, episode_adv: torch.Tensor,
                            step_adv: torch.Tensor, response_mask: torch.Tensor,
                            step_advantage_w: float, extra: Optional[str] = None) -> None:
    """One console line per update showing whether the step term is doing anything.

    If step groups are mostly singletons the step advantage is ~0 and GiGPO reduces to GRPO.
    verl does not configure the root logger, so this prints (like verl's own diagnostics).
    """
    try:
        counts: Dict[Any, int] = defaultdict(int)
        for g in step_group_uids.tolist():
            counts[g] += 1
        sizes = list(counts.values())
        n = sum(sizes)
        non_singleton = 100.0 * sum(s for s in sizes if s >= 2) / max(1, n)
        resp = response_mask.bool()
        ep_mag = float(episode_adv[resp].abs().mean()) if resp.any() else 0.0
        st_mag = float(step_adv[resp].abs().mean()) if resp.any() else 0.0
        nz = 100.0 * float((step_adv[resp].abs() > 1e-8).float().mean()) if resp.any() else 0.0
        print(
            f"[{tag}] {extra + ' | ' if extra else ''}"
            f"step-groups={len(sizes)} over {n} steps non-singleton={non_singleton:.1f}% "
            f"avg_size={n / max(1, len(sizes)):.2f} max={max(sizes) if sizes else 0} | "
            f"|episode_adv|={ep_mag:.4f} w*|step_adv|={step_advantage_w * st_mag:.4f} "
            f"step/episode={(step_advantage_w * st_mag) / (ep_mag + 1e-8):.2f} nonzero-step={nz:.1f}%",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 — diagnostics must never break training
        print(f"[{tag}] diagnostics failed: {e}", flush=True)
