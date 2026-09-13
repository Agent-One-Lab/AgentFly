"""Stateless result-to-training conversion, selected by ``RunResult.rollout``.

Both exporters share segment tokenization and outcome broadcasting. Only their
runtime-step projections and trainer-specific fields differ. No converter reads
the agent's last-run caches or constructs a rollout strategy.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from collections import Counter

from ...utils.verl import pad_tensor_batch_dim_with_zeros, pad_tensor_to_rank_size
from ..types import RunResult
from ..utils.action_format import action_format_valid, message_text
from ..utils.tokenizer import tokenize_trajectories


def result_to_dataproto(
    agent,
    run_result: RunResult,
    *,
    train_on_last_turn: bool = False,
    world_size: int = 1,
    pad_to_multiple_of: Optional[int] = None,
    log_token_drift: bool = True,
):
    """Convert an explicit in-memory result using its producer identifier.

    An unmarked or unsupported result is rejected rather than inferred from its
    segments/steps or from whichever rollout most recently ran on the agent.
    The agent supplies tokenization configuration, not rollout execution state.
    """
    if not isinstance(run_result, RunResult):
        raise TypeError("Expected a RunResult returned by agent.run().")
    converters = {"chain": chain_to_dataproto, "step": step_to_dataproto}
    if run_result.rollout not in converters:
        raise ValueError(
            "RunResult.rollout must identify a supported converter ('chain' or 'step'); "
            f"got {run_result.rollout!r}."
        )
    return converters[run_result.rollout](
        agent, run_result,
        train_on_last_turn=train_on_last_turn,
        world_size=world_size,
        pad_to_multiple_of=pad_to_multiple_of,
        log_token_drift=log_token_drift,
    )


def build_segment_batch(
    agent,
    run_result: RunResult,
    *,
    tokenize: Callable,
    train_on_last_turn: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    missing_reward: Optional[float] = None,
    missing_metric: Any = None,
    metric_dtype=None,
) -> Tuple[Dict[str, Any], List[Tuple[int, int]], List[int], int]:
    """Select training rows, broadcast outcomes/metrics, and align the batch.

    Returns ``(inputs, row_indices, repeat_times, pad_size)``. Each row index is
    ``(trajectory_index, segment_index)``; padding repeats the last real index.
    Indices refer to the original, unmodified result even when segments are
    skipped. Exporters use this mapping for their additional training signals.
    Empty/context-only segments are skipped before tokenization; rows with no
    action-mask tokens after tokenization are skipped too.

    ``tokenize`` is the existing tokenization hook (including overrides), not a
    rollout object. The missing-value and dtype arguments preserve the exporters'
    current boundary behavior; they do not select a rollout or reward layout.
    Every real segment receives the full trajectory outcome, never a fraction.
    """
    import torch  # lazy: importing rollout strategies must not load the trainer

    trajectories = run_result.trajectories
    row_indices = [
        (batch_idx, segment_idx)
        for batch_idx, trajectory in enumerate(trajectories)
        for segment_idx, segment in enumerate(trajectory.segments)
        if any(message.get("role") == "assistant" for message in segment.messages)
    ]
    if not row_indices:
        raise ValueError("No trainable segments: the result contains no assistant messages.")
    conversations = [trajectories[b].segments[s].messages for b, s in row_indices]
    inputs = dict(tokenize(
        agent,
        messages_list=conversations,
        tokenizer=agent.tokenizer,
        processor=agent.processor,
        return_reward_mask=True,
        concatenate_mm_inputs=False,
        train_on_last_turn=train_on_last_turn,
    ))
    if inputs["input_ids"].shape[0] != len(row_indices):
        raise ValueError("Segment tokenization must return one row per segment.")
    if "mm_inputs" in inputs and len(inputs["mm_inputs"]) != len(row_indices):
        raise ValueError("Multimodal metadata must have one entry per segment.")

    # An assistant message alone does not guarantee a training target: masking
    # or max-length truncation can leave a row with no policy tokens. Select
    # tensors and per-row multimodal metadata together, without mutating either
    # tokenizer-owned outputs or the original trajectory/step records.
    action_mask = inputs["action_mask"]
    if action_mask.ndim != 2 or action_mask.shape[0] != len(row_indices):
        raise ValueError("Tokenization must return one action-mask row per segment.")
    kept = action_mask.bool().any(dim=-1).nonzero(as_tuple=True)[0].tolist()
    if not kept:
        raise ValueError("No trainable segments: tokenization produced no action-mask tokens.")
    if len(kept) != len(row_indices):
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value[kept]
            elif key == "mm_inputs":
                inputs[key] = [value[i] for i in kept]
        row_indices = [row_indices[i] for i in kept]

    outcomes = [
        float(trajectories[b].reward if trajectories[b].reward is not None else missing_reward)
        for b, _ in row_indices
    ]
    inputs["rm_scores"] = inputs["reward_mask"] * torch.tensor(
        outcomes, dtype=torch.float32,
    ).unsqueeze(dim=-1)

    repeat_times = [0] * len(trajectories)
    for batch_idx, _ in row_indices:
        repeat_times[batch_idx] += 1
    align = pad_to_multiple_of
    pad_size = (align - len(row_indices) % align) % align if align and align > 1 else 0
    if pad_size:
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                if key == "action_mask" and not value.is_nested:
                    inputs[key] = pad_tensor_batch_dim_with_zeros(value, align)
                else:
                    inputs[key] = pad_tensor_to_rank_size(value, align)
        # chat-bricks returns per-row multimodal dictionaries with
        # concatenate_mm_inputs=False; keep them aligned with repeated token rows.
        if "mm_inputs" in inputs:
            mm_inputs = inputs["mm_inputs"]
            inputs["mm_inputs"] = list(mm_inputs) + [mm_inputs[-1]] * pad_size
        # The final trajectory can have no training rows. Padding belongs to
        # the trajectory whose last real row we actually repeat, not the final
        # entry in RunResult (the trainer uses these counts to repeat task data).
        repeat_times[row_indices[-1][0]] += pad_size
        row_indices = row_indices + [row_indices[-1]] * pad_size

    row_trajectories = [trajectories[b] for b, _ in row_indices]
    inputs["uid"] = np.array([t.group_id for t in row_trajectories], dtype=object)
    metric_keys = sorted({key for t in trajectories for key in t.metrics})
    for key in metric_keys:
        inputs[f"rm_{key}"] = np.array(
            [t.metrics.get(key, missing_metric) for t in row_trajectories],
            dtype=metric_dtype,
        )
    return inputs, row_indices, repeat_times, pad_size


def chain_to_dataproto(
    agent,
    run_result: RunResult,
    *,
    train_on_last_turn: bool = False,
    world_size: int = 1,
    pad_to_multiple_of: Optional[int] = None,
    tokenize: Optional[Callable] = None,
    log_token_drift: bool = True,
):
    """Convert ``run_result``'s trajectories into a verl ``DataProto`` (one row per
    trainable **segment**; context-only/empty views remain in the result).

    See :meth:`agentfly.agents.agent_base.BaseAgent.to_verl_dataproto` for the
    full ``batch`` / ``non_tensor_batch`` / ``meta_info`` contract this produces.
    """
    import torch  # lazy: heavy import, only needed when building tensors
    from ...verl.protocol import DataProto  # lazy: pulls the verl/torch/ray stack

    tokenize = tokenize or tokenize_trajectories
    trajectories = run_result.trajectories
    inputs, row_indices, repeat_times, pad_size = build_segment_batch(
        agent,
        run_result,
        tokenize=tokenize,
        train_on_last_turn=train_on_last_turn,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    n_real = len(row_indices) - pad_size
    drift = None
    if log_token_drift:
        try:
            # Same diagnostic as the step path: spliced (sampled) ids vs a fresh
            # text re-tokenization, on real rows only.
            conversations = [
                trajectories[b].segments[s].messages for b, s in row_indices[:n_real]
            ]
            drift = token_drift_stats(agent, conversations, inputs, tokenize=tokenize)
        except Exception as e:  # noqa: BLE001 — diagnostics must never break the run
            print(f"[ChainRollout] token-drift diagnostic failed: {e}", flush=True)

    # Per-turn rollout signals projected from each row's trajectory ``steps``
    # (turn order; each Step carries its ToolResult). Only these clean arrays
    # cross the boundary; the Step objects stay agent-side. Consumed by
    # step-level estimators (GiGPO); ignored otherwise.
    #
    # ANCHOR = the state the agent acted FROM (pre-action s_t). GiGPO groups
    # turns by the state the action was taken from; ``tool_result.anchor`` is the
    # state AFTER action t (post-action s_{t+1}), so we shift the anchors right by
    # one — anchor[t] = the previous turn's post-state, with a shared marker for
    # the first turn (all chains of a prompt share s_0, and grouping is scoped
    # within uid). ``step_reward[t]`` is the reward from action t, already aligned.
    step_observations_list = []
    step_rewards_list = []
    step_invalids_list = []
    for batch_idx, _ in row_indices:
        steps = trajectories[batch_idx].steps
        # Prefer the raw ``anchor`` (undecorated state key) over the LLM-facing
        # ``observation`` (which may carry an admissible-action menu that fragments
        # exact-hash grouping); fall back to ``observation`` when no anchor is set.
        post_obs = [
            s.tool_result.anchor
            if s.tool_result is not None and s.tool_result.anchor is not None
            else s.observation
            for s in steps
        ]
        step_observations_list.append((["__init__"] + post_obs[:-1]) if post_obs else [])
        # ``step_reward[t]`` is the RAW env reward from action t (already aligned).
        step_rewards_list.append(
            [s.tool_result.step_reward if s.tool_result is not None else None for s in steps]
        )
        # ``step_invalids[t]`` = whether action t was FORMAT-invalid — verl-agent's
        # ``alfworld_projection`` check on the model's raw output (``<action>`` +
        # ``<think>`` + no Chinese), the same definition StepRollout uses — so the GiGPO
        # estimator applies the invalid-action penalty as a local, post-discount
        # per-step deduction (matching verl-agent). NOT the env-semantic "nothing
        # happens" flag (which fires constantly and gets amplified to ~-sqrt(N)), and a
        # no-tool-call turn is invalid here too (``Step.is_action_valid`` would say 1.0).
        step_invalids_list.append(
            [
                1.0 - action_format_valid(message_text(s.messages.messages[-1]))
                if s.messages.messages
                else 1.0
                for s in steps
            ]
        )
    # For discarded trajectories, mask out all response tokens for every segment row.
    discarded = [
        bool(t.model_dump(exclude={"segments"}).get("discarded", False))
        for t in trajectories
    ]
    if any(discarded):
        discarded_tensor = torch.tensor(
            [discarded[b] for b, _ in row_indices],
            dtype=torch.bool, device=inputs["attention_mask"].device,
        ).unsqueeze(dim=-1)
        for key in ("action_mask", "reward_mask", "rm_scores"):
            if key in inputs:
                inputs[key] = inputs[key] * (~discarded_tensor).to(dtype=inputs[key].dtype)

    inputs["batch_idx"] = np.array([b for b, _ in row_indices], dtype=np.int32)
    inputs["segment_idx"] = np.array([s for _, s in row_indices], dtype=np.int32)
    # 1D object arrays of per-turn lists (np.empty avoids equal-length rows
    # collapsing into a 2D array).
    step_observations = np.empty(len(step_observations_list), dtype=object)
    step_observations[:] = step_observations_list
    step_rewards = np.empty(len(step_rewards_list), dtype=object)
    step_rewards[:] = step_rewards_list
    step_invalids = np.empty(len(step_invalids_list), dtype=object)
    step_invalids[:] = step_invalids_list
    inputs["step_observations"] = step_observations
    inputs["step_rewards"] = step_rewards
    inputs["step_invalids"] = step_invalids

    if "mm_inputs" in inputs:
        mm_inputs = inputs.pop("mm_inputs")
        inputs["multi_modal_inputs"] = np.array(mm_inputs, dtype=object)
    try:
        won = sum(1 for t in trajectories if (t.reward or 0.0) > 0)
        finishes = Counter(t.finish_reason for t in trajectories)
        drift_str = ""
        if drift:
            drift_str = " token_drift={" + ", ".join(
                f"{k.split('/', 1)[1]}={v:.4f}" for k, v in sorted(drift.items())
            ) + "}"
        # Loop-policy counters carried in Trajectory.metadata (see ChainRollout):
        # turns that hit the per-turn token cap, tool-less turns, and nudges sent.
        def _meta_sum(key):
            return sum(int((t.metadata or {}).get(key, 0) or 0) for t in trajectories)
        capped, no_tool, nudges = (
            _meta_sum("capped_turns"), _meta_sum("no_tool_call_turns"), _meta_sum("no_tool_call_nudges"),
        )
        print(
            f"[ChainRollout] batch: rows={n_real} (+{pad_size} pad) trajs={len(trajectories)} "
            f"won={won}/{len(trajectories)} finish={dict(finishes)} "
            f"capped_turns={capped} no_tool_call_turns={no_tool} nudges={nudges}{drift_str}",
            flush=True,
        )
        rollout_stats = {"capped_turns": capped, "no_tool_call_turns": no_tool, "no_tool_call_nudges": nudges}
    except Exception as e:  # noqa: BLE001 — debug print must never break the run
        print(f"[ChainRollout] batch summary failed: {e}", flush=True)
    meta_info = {"use_agent": True, "layout": "per_segment", "repeat_times": repeat_times}
    if drift:
        meta_info["token_drift"] = drift
    try:
        meta_info["rollout_stats"] = rollout_stats
    except NameError:  # summary block failed before computing the counters
        pass
    batch = DataProto.from_single_dict(inputs, meta_info=meta_info)
    return batch


def step_to_dataproto(
    agent,
    run_result: RunResult,
    *,
    train_on_last_turn: bool = False,
    world_size: int = 1,
    pad_to_multiple_of: Optional[int] = None,
    tokenize: Optional[Callable] = None,
    log_token_drift: bool = True,
    token_drift: Optional[Callable] = None,
):
    """Flatten ``run_result.trajectories`` into a verl ``DataProto`` — ONE ROW PER STEP.

    ``batch``: ``input_ids``/``attention_mask``/``position_ids``/``action_mask``
    (per step), ``rm_scores`` = the trajectory **outcome** broadcast to every
    step-row (on the reward-mask token, scale 1.0).
    ``non_tensor_batch``: ``uid`` (task group), ``traj_uid``, ``anchor_obs``
    (the pre-action state, right-shifted like ChainRollout), ``step_env_reward``,
    ``is_action_valid``, ``active_masks`` (0 on divisor-padding rows).
    ``meta_info``: ``layout="per_step"`` — the trainer resolves the estimator's
    ``per_step`` implementation in :mod:`agentfly.algorithms`.
    """
    from ...verl.protocol import DataProto  # lazy: pulls the verl/torch/ray stack

    tokenize = tokenize or tokenize_trajectories

    # The step-only invariant and pre-action anchors remain in this exporter.
    # Shared conversion uses segments, never these runtime records.
    pre_anchors = []
    for traj in run_result.trajectories:
        steps = traj.steps
        if not steps and not any(
            message.get("role") == "assistant"
            for segment in traj.segments for message in segment.messages
        ):
            # A zero-generation rollout can retain an empty/context-only view.
            # It contributes no training rows and needs no runtime-step mapping.
            pre_anchors.append([])
            continue
        if len(steps) != traj.num_segments:
            raise ValueError(
                "Step rollout conversion requires one runtime step per segment; "
                f"got {len(steps)} steps and {traj.num_segments} segments "
                f"for trajectory {traj.chain_id!r}."
            )
        post = [
            s.tool_result.anchor
            if s.tool_result is not None and s.tool_result.anchor is not None
            else s.observation
            for s in steps
        ]
        pre_anchors.append((["__init__"] + post[:-1]) if post else [])

    inputs, row_indices, repeat_times, pad_size = build_segment_batch(
        agent,
        run_result,
        tokenize=tokenize,
        train_on_last_turn=train_on_last_turn,
        pad_to_multiple_of=pad_to_multiple_of,
        # Preserve this exporter's legacy missing-value and metric-dtype rules.
        missing_reward=0.0,
        missing_metric=0.0,
        metric_dtype=object,
    )
    row_traj = [run_result.trajectories[b] for b, _ in row_indices]
    row_segment = [run_result.trajectories[b].segments[s] for b, s in row_indices]
    row_step = [run_result.trajectories[b].steps[s] for b, s in row_indices]
    row_anchor = [pre_anchors[b][s] for b, s in row_indices]
    n_real = len(row_indices) - pad_size

    drift = None
    if log_token_drift:
        try:
            # Diagnostics sample real rows only, never divisor-padding copies.
            conversations = [segment.messages for segment in row_segment[:n_real]]
            if token_drift is None:
                drift = token_drift_stats(agent, conversations, inputs, tokenize=tokenize)
            else:
                drift = token_drift(agent, conversations, inputs)
        except Exception as e:  # noqa: BLE001 — diagnostics must never break the run
            print(f"[StepRollout] token-drift diagnostic failed: {e}", flush=True)

    inputs["traj_uid"] = np.array([t.chain_id for t in row_traj], dtype=object)
    anchors = np.empty(len(row_anchor), dtype=object)
    anchors[:] = row_anchor
    inputs["anchor_obs"] = anchors
    inputs["step_env_reward"] = np.array(
        [
            float((s.tool_result.step_reward if s.tool_result is not None else None) or 0.0)
            for s in row_step
        ],
        dtype=np.float32,
    )
    # ``is_action_valid`` is verl-agent's FORMAT check on the model's raw output
    # (``alfworld_projection``): the response must carry a ``<action>...</action>`` block AND
    # a ``<think>...</think>`` block AND no Chinese characters. It is deliberately NOT the
    # env-semantic "nothing happens" flag (which stays a diagnostic reward metric): verl-agent
    # penalizes malformed *output*, not failed *actions*. The env flag fires on nearly every
    # turn, and in the dominant zero-win groups ``mean_std_norm`` amplifies a lone
    # invalid-action penalty to ~-sqrt(group_size) (the observed -19.946 = -sqrt(~398) noise
    # floor), drowning the win signal. A no-tool-call turn has no ``<action>`` and so is
    # format-invalid here as well (subsumes the earlier tool_result-None override).
    inputs["is_action_valid"] = np.array(
        [
            action_format_valid(message_text(segment.messages[-1]))
            if segment.messages
            else 0.0
            for segment in row_segment
        ],
        dtype=np.float32,
    )
    # Real vs divisor-padding rows: the per-step GiGPO estimator restricts grouping /
    # discounting to active rows (padded rows repeat the last row's non-tensors, which
    # would otherwise corrupt the trajectory/anchor groups).
    inputs["active_masks"] = np.array(
        [1.0] * n_real + [0.0] * pad_size, dtype=np.float32
    )

    # --- debug: per-step batch summary (rows, trajectories, outcomes, lengths) ---
    try:
        from collections import Counter
        trajs = run_result.trajectories
        n_traj = len(trajs)
        won = sum(1 for t in trajs if (t.reward or 0.0) > 0)
        finishes = Counter(t.finish_reason for t in trajs)
        lens = [0] * n_traj
        for batch_idx, _ in row_indices[:n_real]:
            lens[batch_idx] += 1
        drift_str = ""
        if drift:
            drift_str = " token_drift={" + ", ".join(
                f"{k.split('/', 1)[1]}={v:.4f}" for k, v in sorted(drift.items())
            ) + "}"
        # Format-valid fraction over real rows (verl-agent's ``valid_action_ratio``).
        valid_frac = float(np.mean(inputs["is_action_valid"][:n_real])) if n_real else 0.0
        print(
            f"[StepRollout] per_step batch: rows={n_real} (+{pad_size} pad) "
            f"trajs={n_traj} training_rows/traj[min/mean/max]="
            f"{min(lens)}/{sum(lens)/max(1,n_traj):.1f}/{max(lens)} "
            f"won={won}/{n_traj} ({100.0*won/max(1,n_traj):.1f}%) "
            f"valid_frac={valid_frac:.3f} finish={dict(finishes)}"
            f"{drift_str}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 — debug print must never break the run
        print(f"[StepRollout] batch summary failed: {e}", flush=True)

    meta_info = {"use_agent": True, "layout": "per_step", "repeat_times": repeat_times}
    if drift:
        meta_info["token_drift"] = drift
    return DataProto.from_single_dict(inputs, meta_info=meta_info)


def token_drift_stats(
    agent, conversations, inputs, max_rows: int = 64, *, tokenize: Optional[Callable] = None,
):
    """Diagnostic: how much the spliced (sampled) assistant ids differ from what
    re-tokenizing the decoded text would have produced — i.e. the retokenization
    drift that training silently absorbed before ``token_ids`` were carried.

    For up to ``max_rows`` rows whose assistant turn carries ``token_ids``, compare
    the action-masked span of the real ``inputs`` (spliced) against a fresh
    tokenization of the same conversation with ``token_ids`` removed (pure text
    path). ``frac_tokens_differ`` is ``1 - SequenceMatcher.ratio()`` (an
    edit-similarity, robust to length changes), averaged over sampled rows.
    Also reports the fraction of carried ids ending in the tokenizer's eos.
    """
    import difflib

    tokenize = tokenize or tokenize_trajectories

    def _last_ids(conv):
        for m in reversed(conv):
            if m.get("role") == "assistant" and m.get("token_ids"):
                return m["token_ids"]
        return None

    n = len(conversations)
    with_ids = [i for i in range(n) if _last_ids(conversations[i])]
    stats = {"token_drift/frac_rows_with_ids": len(with_ids) / max(1, n)}
    if not with_ids:
        return stats

    eos_id = getattr(getattr(agent, "tokenizer", None), "eos_token_id", None)
    if eos_id is not None:
        ends = [_last_ids(conversations[i])[-1] == eos_id for i in with_ids]
        stats["token_drift/frac_ids_end_with_eos"] = sum(ends) / len(ends)

    sample = with_ids[:max_rows]
    stripped = [
        [{k: v for k, v in m.items() if k != "token_ids"} for m in conversations[i]]
        for i in sample
    ]
    text_inputs = tokenize(
        agent,
        messages_list=stripped,
        tokenizer=agent.tokenizer,
        processor=agent.processor,
        return_reward_mask=False,
        concatenate_mm_inputs=False,
        train_on_last_turn=False,
    )
    rows_differ, tok_diff = 0, []
    for j, i in enumerate(sample):
        a = inputs["input_ids"][i][inputs["action_mask"][i] == 1].tolist()
        b = text_inputs["input_ids"][j][text_inputs["action_mask"][j] == 1].tolist()
        rows_differ += int(a != b)
        tok_diff.append(
            1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
        )
    stats["token_drift/frac_rows_differ"] = rows_differ / len(sample)
    stats["token_drift/frac_tokens_differ"] = sum(tok_diff) / len(sample)
    stats["token_drift/n_sampled"] = float(len(sample))
    return stats
