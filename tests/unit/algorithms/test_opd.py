"""OPD: teacher step (alignment, validation, client reuse) and the OPD estimator (think spans,
clamped log-ratio advantage, GRPO term, layouts, metrics)."""

import asyncio
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiohttp import web

from agentfly import algorithms
from agentfly.algorithms import opd
from agentfly.algorithms.opd import (
    OPD_DEFAULTS,
    THINK_END_TOKEN_ID as END,
    compute_opd_advantage,
    compute_opd_metrics,
    opd_options,
    think_position_mask,
)
from agentfly.algorithms.teacher import ScoreStats


def right_padded(rows, width):
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row)
        mask[i, : len(row)] = 1
    return SimpleNamespace(batch={"input_ids": ids, "attention_mask": mask})


class TokenIdTeacher:
    """Fake client: log p(token) = token id, so outputs reveal exactly which token was placed where."""

    def __init__(self):
        self.calls = []

    def score_blocking(self, sequences):
        self.calls.append([list(s) for s in sequences])
        logprobs = []
        for seq in sequences:
            lp = np.array(seq, dtype=np.float32)
            lp[0] = np.nan
            logprobs.append(lp)
        return logprobs, ScoreStats(sequences=len(sequences), unique=len({tuple(s) for s in sequences}),
                                    tokens=sum(len(s) for s in sequences), requests=len(sequences))


@pytest.fixture
def fake_teacher(monkeypatch):
    fake = TokenIdTeacher()
    monkeypatch.setattr(opd, "teacher_client", lambda config: fake)
    return fake


def test_teacher_log_prob_at_t_is_for_the_next_token(fake_teacher):
    batch = right_padded([[11, 12, 13, 14], [21, 22], [31, 32, 33]], width=6)

    out, metrics = opd.compute_teacher_log_probs(batch, {"endpoints_file": "unused"})

    assert out.dtype == torch.float32 and out.shape == (3, 6)
    expected = torch.tensor([
        [12, 13, 14, 0, 0, 0],   # t < n-1 holds token t+1; last valid position and padding are 0
        [22, 0, 0, 0, 0, 0],
        [32, 33, 0, 0, 0, 0],
    ], dtype=torch.float32)
    torch.testing.assert_close(out, expected)
    assert fake_teacher.calls == [[[11, 12, 13, 14], [21, 22], [31, 32, 33]]]  # padding never sent
    assert metrics["teacher/rows"] == 3 and metrics["teacher/tokens"] == 9


def test_matches_old_log_probs_convention_under_the_shifted_action_mask(fake_teacher):
    # The trainer shifts action_mask left by one: position t is trained when token t+1 is generated.
    ids = [5, 6, 7, 8, 9]
    generated = torch.tensor([[0, 0, 1, 1, 0, 0]])           # tokens 7 and 8 were sampled
    shifted = torch.zeros_like(generated)
    shifted[:, :-1] = generated[:, 1:]
    out, _ = opd.compute_teacher_log_probs(right_padded([ids], width=6), {"endpoints_file": "x"})
    assert out[shifted.bool()].tolist() == [7.0, 8.0]


def test_single_token_rows_and_full_width_rows(fake_teacher):
    out, _ = opd.compute_teacher_log_probs(right_padded([[4], [1, 2, 3]], width=3), {"endpoints_file": "x"})
    torch.testing.assert_close(out, torch.tensor([[0, 0, 0], [2, 3, 0]], dtype=torch.float32))


def test_left_padding_is_rejected(fake_teacher):
    ids = torch.tensor([[0, 0, 7, 8]])
    mask = torch.tensor([[0, 0, 1, 1]])
    with pytest.raises(ValueError, match="right-padded"):
        opd.compute_teacher_log_probs(SimpleNamespace(batch={"input_ids": ids, "attention_mask": mask}),
                                      {"endpoints_file": "x"})
    assert fake_teacher.calls == []


def test_mismatched_shapes_are_rejected(fake_teacher):
    batch = SimpleNamespace(batch={"input_ids": torch.zeros(2, 4, dtype=torch.long),
                                   "attention_mask": torch.ones(2, 3, dtype=torch.long)})
    with pytest.raises(ValueError, match="same"):
        opd.compute_teacher_log_probs(batch, {"endpoints_file": "x"})


def test_endpoints_file_is_required():
    with pytest.raises(ValueError, match="endpoints_file is required"):
        opd.teacher_client({})


def test_client_is_created_once_per_settings(monkeypatch):
    monkeypatch.setattr(opd, "_CLIENTS", {})
    a = opd.teacher_client({"endpoints_file": "/tmp/t.json", "expect_model": "m"})
    b = opd.teacher_client({"endpoints_file": "/tmp/t.json", "expect_model": "m"})
    c = opd.teacher_client({"endpoints_file": "/tmp/t.json", "expect_model": "other"})
    assert a is b and a is not c
    assert a.expect_model == "m" and a.max_wait_s == 1800.0


def test_end_to_end_through_the_real_client(tmp_path, monkeypatch):
    """Real TeacherClient over HTTP: alignment holds and duplicate rows reach the server once."""
    prompts = []

    async def completions(request):
        body = await request.json()
        prompts.append(body["prompt"])
        entries = [None] + [{str(t): {"logprob": -float(t)}} for t in body["prompt"][1:]]
        return web.json_response({"choices": [{"prompt_logprobs": entries}]})

    async def models(request):
        return web.json_response({"data": [{"id": "teacher"}]})

    loop, ready, holder = asyncio.new_event_loop(), threading.Event(), {}

    def serve():
        asyncio.set_event_loop(loop)

        async def start():
            app = web.Application()
            app.router.add_post("/v1/completions", completions)
            app.router.add_get("/v1/models", models)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            holder["runner"], holder["port"] = runner, site._server.sockets[0].getsockname()[1]

        loop.run_until_complete(start())
        ready.set()
        loop.run_forever()

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    assert ready.wait(10)
    try:
        endpoints = tmp_path / "teacher.json"
        endpoints.write_text(json.dumps({
            "name": "teacher", "model": "org/Teacher", "max_model_len": 100, "status": "ready",
            "urls": [f"http://127.0.0.1:{holder['port']}"], "dp_size": 1,
        }))
        monkeypatch.setattr(opd, "_CLIENTS", {})
        batch = right_padded([[3, 4, 5], [7, 8], [3, 4, 5]], width=4)   # row 2 repeats row 0 (padding copy)

        out, metrics = opd.compute_teacher_log_probs(batch, {"endpoints_file": str(endpoints),
                                                             "expect_model": "org/Teacher"})
    finally:
        asyncio.run_coroutine_threadsafe(holder["runner"].cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        server.join(10)

    torch.testing.assert_close(out, torch.tensor([[-4, -5, 0, 0], [-8, 0, 0, 0], [-4, -5, 0, 0]],
                                                 dtype=torch.float32))
    assert sorted(map(tuple, prompts)) == [(3, 4, 5), (7, 8)]
    assert metrics["teacher/rows"] == 3 and metrics["teacher/unique_rows"] == 2


# ---- OPD estimator -----------------------------------------------------------------------------

def shifted(token_mask):
    """The trainer's position mask: position t is trained when token t+1 was generated."""
    positions = torch.zeros_like(token_mask)
    positions[:, :-1] = token_mask[:, 1:]
    return positions


def multi_turn_rows():
    """Row 0: 3 turns (think+answer, think+answer, cut off mid-thinking) with observations between.
    Row 1: a turn with empty thinking (first token </think>), then a turn with two </think> tokens."""
    ids = torch.tensor([
        [1, 2, 10, 11, END, 20, 21, 3, 4, 30, END, 40, 5, 6, 50, 51, 0],
        [1, END, 60, 61, 7, 70, END, 71, END, 72, 0, 0, 0, 0, 0, 0, 0],
    ])
    generated = torch.tensor([
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ])
    think_tokens = torch.tensor([
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],   # up to and including the first </think>
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # only the first </think> ends thinking
    ])
    return ids, shifted(generated), shifted(think_tokens).bool()


def opd_batch(ids, action_mask, teacher, old, **fields):
    return SimpleNamespace(
        batch={"input_ids": ids, "action_mask": action_mask, "teacher_log_probs": teacher,
               "old_log_probs": old, **{k: v for k, v in fields.items() if k != "uid"}},
        non_tensor_batch={"uid": fields["uid"]} if "uid" in fields else {},
        meta_info={"layout": "per_segment"},
    )


def test_think_positions_follow_turns_in_the_shifted_convention():
    ids, action_mask, expected = multi_turn_rows()
    torch.testing.assert_close(think_position_mask(ids, action_mask), expected)


def test_advantage_is_masked_clamped_log_ratio():
    torch.manual_seed(0)
    ids, action_mask, _ = multi_turn_rows()
    teacher, old = torch.randn(ids.shape) * 3, torch.randn(ids.shape) * 3
    teacher[0, 2], old[0, 2] = 10.0, 0.0          # ratio +10 -> clamp
    teacher[0, 3], old[0, 3] = -12.0, 0.0         # ratio -12 -> clamp
    advantages, returns = compute_opd_advantage(opd_batch(ids, action_mask, teacher, old), {}, "per_segment")
    expected = action_mask * torch.clamp(teacher - old, -5.0, 5.0)
    torch.testing.assert_close(advantages, expected)
    assert advantages[0, 2] == 5.0 and advantages[0, 3] == -5.0
    assert torch.equal(advantages[action_mask == 0], torch.zeros_like(advantages[action_mask == 0]))
    assert returns is advantages


def test_think_weight_scales_only_thinking_positions():
    ids, action_mask, think = multi_turn_rows()
    teacher = torch.full(ids.shape, 1.0)
    old = torch.zeros(ids.shape)
    config = {"opd": {"think_weight": 0.3}}
    advantages, _ = compute_opd_advantage(opd_batch(ids, action_mask, teacher, old), config, "per_segment")
    trained = action_mask > 0
    assert torch.allclose(advantages[think], torch.full_like(advantages[think], 0.3))
    assert torch.allclose(advantages[trained & ~think], torch.ones_like(advantages[trained & ~think]))


def test_task_reward_term_adds_verls_grpo_advantage():
    from agentfly.verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    torch.manual_seed(1)
    ids = torch.ones(4, 6, dtype=torch.long)
    action_mask = torch.tensor([[0, 1, 1, 1, 0, 0]] * 4, dtype=torch.float32)
    rewards = torch.zeros(4, 6)
    rewards[:, 3] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    uid = np.array(["a", "a", "b", "b"], dtype=object)
    teacher, old = torch.randn(4, 6), torch.randn(4, 6)
    batch = opd_batch(ids, action_mask, teacher, old, token_level_rewards=rewards, uid=uid)

    pure, _ = compute_opd_advantage(batch, {}, "per_segment")
    hybrid, _ = compute_opd_advantage(batch, {"opd": {"task_reward_coef": 0.5}}, "per_segment")
    grpo, _ = compute_grpo_outcome_advantage(token_level_rewards=rewards, response_mask=action_mask, index=uid)
    torch.testing.assert_close(hybrid, pure + 0.5 * grpo * action_mask)


def test_registered_for_both_layouts_with_identical_results():
    torch.manual_seed(2)
    ids, action_mask, _ = multi_turn_rows()
    batch = opd_batch(ids, action_mask, torch.randn(ids.shape), torch.randn(ids.shape))
    assert algorithms.estimator_layouts("opd") == ["per_segment", "per_step"]
    segment, _ = algorithms.get_estimator("opd", "per_segment")(batch, {})
    step, _ = algorithms.get_estimator("opd", "per_step")(batch, {})
    torch.testing.assert_close(segment, step)
    assert algorithms.estimator_layouts("gigpo") == ["per_segment", "per_step"]


def test_missing_teacher_log_probs_names_the_field():
    ids, action_mask, _ = multi_turn_rows()
    batch = opd_batch(ids, action_mask, None, torch.zeros(ids.shape))
    del batch.batch["teacher_log_probs"]
    with pytest.raises(KeyError, match="teacher_log_probs"):
        compute_opd_advantage(batch, {}, "per_segment")


def test_options_defaults_and_overrides():
    assert opd_options({}) == OPD_DEFAULTS == {"clamp": 5.0, "think_weight": 1.0, "task_reward_coef": 0.0}
    assert opd_options({"opd": {"clamp": 2}}) == {**OPD_DEFAULTS, "clamp": 2.0}


def test_metrics_split_kl_by_thinking():
    ids, action_mask, think = multi_turn_rows()
    trained = action_mask > 0
    old = torch.zeros(ids.shape)
    teacher = torch.where(think, torch.tensor(-1.0), torch.tensor(-0.5))   # k1 = 1.0 thinking, 0.5 else
    teacher[0, 5] = 7.0                                                    # one non-thinking outlier
    batch = opd_batch(ids, action_mask, teacher, old)
    batch.batch["advantages"], _ = compute_opd_advantage(batch, {}, "per_segment")
    metrics = compute_opd_metrics(batch, {})

    k1 = (old - teacher)[trained]
    non_think = trained & ~think
    assert metrics["opd/kl_k1_mean"] == pytest.approx(float(k1.mean()))
    assert metrics["opd/kl_k1_think"] == pytest.approx(1.0)
    assert metrics["opd/kl_k1_non_think"] == pytest.approx(float((old - teacher)[non_think].mean()))
    assert metrics["opd/think_token_frac"] == pytest.approx(float(think.sum() / trained.sum()))
    assert metrics["opd/teacher_prefers_frac"] == pytest.approx(1 / float(trained.sum()))
    assert metrics["opd/adv_clamp_frac"] == pytest.approx(1 / float(trained.sum()))
    assert metrics["opd/adv_abs_mean"] == pytest.approx(
        float(batch.batch["advantages"][trained].abs().mean()))
