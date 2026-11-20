"""The rollout engine's log-probabilities and routed-expert rows, carried per turn and aligned with training rows
(``agents.rollout.loop.routing``), and the per-step parity metric."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agentfly.agents.rollout.loop.routing import (  # noqa: E402
    LOGPROBS,
    PAD,
    ROUTES,
    RoutedRows,
    assemble_batch,
    assemble_row,
    attach_rollout_signals,
    digest,
    held_rows,
    sync_rollout_signals,
)
from agentfly.algorithms.rollout_parity import rollout_parity_metrics  # noqa: E402

LAYERS, K = 3, 2


def engine_rows(start, stop):
    """What the engine returns for positions [start, stop): the position number in every cell."""
    return np.repeat(np.arange(start, stop, dtype=np.uint8)[:, None, None], LAYERS * K, 1).reshape(-1, LAYERS, K)


def turn(prompt, sampled, start):
    """An assistant message as the generation step leaves it, for a request with this prompt."""
    rows = engine_rows(start, len(prompt) + len(sampled) - 1)
    return {"role": "assistant", "content": "x", "token_ids": list(sampled),
            LOGPROBS: [-0.1 * (i + 1) for i in range(len(sampled))],
            ROUTES: RoutedRows(rows, start, len(prompt), digest(prompt))}


def conversation():
    """Two turns: prompt [0..4] + sampled [10, 11, 12]; then context [20, 21] + sampled [30, 31]."""
    p1 = [0, 1, 2, 3, 4]
    t1 = turn(p1, [10, 11, 12], 0)
    p2 = p1 + [10, 11, 12] + [20, 21]
    assert held_rows([{"role": "user"}, t1, {"role": "tool"}], p2) == 7
    t2 = turn(p2, [30, 31], 7)
    ids = np.array(p2 + [30, 31])
    mask = np.array([0] * 5 + [1] * 3 + [0] * 2 + [1] * 2)
    return [{"role": "user"}, t1, {"role": "tool"}, t2], ids, mask


def test_next_request_starts_after_the_rows_already_held():
    messages, ids, _ = conversation()
    assert held_rows(messages, ids.tolist() + [40]) == 11  # 12 tokens so far -> rows 0..10 held


def test_a_changed_prompt_starts_over():
    messages, ids, _ = conversation()
    changed = ids.tolist()
    changed[1] = 99  # an earlier token was re-rendered differently
    assert held_rows(messages, changed) == 0
    assert held_rows(messages[:2], [0, 1, 2, 3, 4, 10, 11]) == 0  # the sampled tokens are not all in the prompt


def test_rows_cover_every_position_but_the_last():
    messages, ids, mask = conversation()
    out = assemble_row(messages, ids, mask)
    assert out["status"] == "ok"
    assert out["routes"].shape == (12, LAYERS, K)
    assert out["routes"][:11, 0, 0].tolist() == list(range(11))  # row p is position p, context included
    assert (out["routes"][11] == PAD).all()
    assert np.allclose(out["log_probs"][5:8], [-0.1, -0.2, -0.3]) and np.allclose(out["log_probs"][10:], [-0.1, -0.2])
    assert np.isnan(out["log_probs"][:5]).all() and np.isnan(out["log_probs"][8:10]).all()


def test_a_later_request_does_not_replace_earlier_rows():
    """Turn 2's request only returns positions it was asked for; turn 1's rows stay as recorded."""
    messages, ids, mask = conversation()
    messages[1][ROUTES].rows[:] = 77
    assert (assemble_row(messages, ids, mask)["routes"][:7] == 77).all()


def test_a_cut_turn_keeps_matching_rows_and_logprobs():
    """A turn kept up to its first two sampled tokens: ids, log-probabilities and rows are cut together."""
    p1 = [0, 1, 2, 3, 4]
    msg = turn(p1, [10, 11, 12, 13], 0)
    msg["token_ids"] = [10, 11]
    sync_rollout_signals([msg])
    assert msg[LOGPROBS] == pytest.approx([-0.1, -0.2]) and len(msg[ROUTES].rows) == 6  # positions 0..5
    assert held_rows([msg], p1 + [10, 11] + [20]) == 6  # position 6 (token 11 as input) comes with the next request
    del msg["token_ids"]
    sync_rollout_signals([msg])
    assert LOGPROBS not in msg and ROUTES not in msg


def test_copies_of_the_conversation_share_the_rows():
    messages, _, _ = conversation()
    assert copy.deepcopy(messages)[1][ROUTES] is messages[1][ROUTES]


def test_attach_from_the_backend_response():
    msgs = [{"role": "assistant", "content": "a", "token_ids": [1, 2]}]
    rows = RoutedRows(engine_rows(0, 3), 0, 2, digest([7, 8]))
    attach_rollout_signals(msgs, {"response_logprobs": [[-1.0, -2.0]], "routed_experts": [rows]})
    assert msgs[0][LOGPROBS] == [-1.0, -2.0] and msgs[0][ROUTES] is rows
    plain = [{"role": "assistant", "content": "a"}]
    attach_rollout_signals(plain, {"response_texts": ["a"]})
    assert LOGPROBS not in plain[0] and ROUTES not in plain[0]


def _inputs(rows):
    width = max(len(ids) for ids, _ in rows)
    pad = lambda v: list(v) + [0] * (width - len(v))  # noqa: E731
    return {"input_ids": torch.tensor([pad(ids) for ids, _ in rows]),
            "action_mask": torch.tensor([pad(m) for _, m in rows]),
            "attention_mask": torch.tensor([pad([1] * len(ids)) for ids, _ in rows])}


def test_batch_marks_rows_whose_ids_do_not_fit_the_record():
    good, ids, mask = conversation()
    bad, bad_ids, _ = conversation()
    bad_ids = bad_ids.copy()
    bad_ids[2] = 99  # the training row differs from what the engine ran
    out = assemble_batch([good, bad], _inputs([(ids, mask), (bad_ids, mask)]), rows_total=3)
    assert out["mismatched"] == [1] and out["statuses"] == {"ok": 1, "prompt_ids_differ": 1}
    routed = out["tensors"]["routed_experts"]
    assert routed.shape == (3, 12, LAYERS, K) and routed.dtype == torch.uint8
    assert routed[0, :11, 0, 0].tolist() == list(range(11)) and (routed[1:] == PAD).all()
    assert torch.isnan(out["tensors"]["engine_log_probs"][2]).all()  # the padding row carries nothing


def test_batch_without_any_record_is_left_alone():
    plain = [{"role": "user"}, {"role": "assistant", "content": "a", "token_ids": [10, 11]}]
    ids, mask = np.array([0, 1, 10, 11]), np.array([0, 0, 1, 1])
    assert assemble_batch([plain], _inputs([(ids, mask)]), rows_total=1) is None


def test_logprobs_alone_are_aligned_by_the_sampled_ids():
    msg = {"role": "assistant", "content": "a", "token_ids": [10, 11], LOGPROBS: [-0.5, -0.25]}
    ids, mask = np.array([0, 1, 10, 11]), np.array([0, 0, 1, 1])
    out = assemble_batch([[{"role": "user"}, msg]], _inputs([(ids, mask)]), rows_total=1)
    assert "routed_experts" not in out["tensors"] and out["mismatched"] == []
    assert out["tensors"]["engine_log_probs"][0, 2:].tolist() == [-0.5, -0.25]


def test_parity_metric_compares_the_same_token():
    """engine_log_probs is indexed by token position, old_log_probs and the shifted action_mask by the state
    before it."""
    engine = torch.full((1, 6), float("nan"))
    engine[0, 3:5] = torch.tensor([-0.5, -1.0])  # tokens 3 and 4 were sampled
    old = torch.zeros(1, 6)
    old[0, 2:4] = torch.tensor([-0.52, -0.97])  # the trainer's values for tokens 3 and 4
    action = torch.zeros(1, 6, dtype=torch.long)
    action[0, 2:4] = 1
    routed = torch.full((1, 6, LAYERS, K), PAD, dtype=torch.uint8)
    routed[0, :3] = 1
    batch = SimpleNamespace(batch={"engine_log_probs": engine, "old_log_probs": old, "action_mask": action,
                                   "attention_mask": torch.ones(1, 6, dtype=torch.long), "routed_experts": routed},
                            meta_info={"rollout_record": {"mismatched": 2}})
    m = rollout_parity_metrics(batch)
    assert m["rollout_parity/logprob_gap_mean"] == pytest.approx(0.025, abs=1e-6)
    assert m["rollout_parity/logprob_gap_signed_mean"] == pytest.approx(0.005, abs=1e-6)
    assert m["rollout_parity/sampled_tokens_scored_by_engine"] == 1.0
    assert m["rollout_parity/positions_with_routes"] == pytest.approx(0.5)
    assert m["rollout_parity/rows_left_out"] == 2.0
