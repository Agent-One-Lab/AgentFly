"""The trainer's OPD hook, compiled from the trainer source without Ray or GPUs.

Covers the teacher step (started on the trainer's background loop, joined before the advantage),
startup checks for adv_estimator=opd, the OPD metrics block, and where fit() calls them.
"""

import ast
import asyncio
import math
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import agentfly.algorithms as agent_algorithms

TRAINER = Path(__file__).resolve().parents[4] / "verl/verl/trainer/ppo/ray_trainer.py"
METHODS = ("_check_agent_teacher_config", "_start_agent_teacher", "_finish_agent_teacher",
           "_agent_algorithm_metrics")


@pytest.fixture(scope="module")
def trainer_source():
    tree = ast.parse(TRAINER.read_text())
    trainer_cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    methods = {n.name: n for n in trainer_cls.body if isinstance(n, ast.FunctionDef)}
    namespace = {"agent_algorithms": agent_algorithms, "asyncio": asyncio, "torch": torch, "math": math,
                 "SimpleNamespace": SimpleNamespace, "DataProto": object}
    module = ast.Module(body=[methods[name] for name in METHODS], type_ignores=[])
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return {"functions": {name: namespace[name] for name in METHODS}, "fit": methods["fit"]}


@pytest.fixture
def background_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)


def make_config(adv_estimator="opd", endpoints_file="/tmp/teacher.json", ppo_epochs=1, use_kl_loss=False,
                entropy_coeff=0.0, ppo_mini_batch_size=4, rollout_n=1):
    return OmegaConf.create({
        "algorithm": {"adv_estimator": adv_estimator,
                      "teacher": {"endpoints_file": endpoints_file, "expect_model": None},
                      "opd": {"clamp": 5.0, "think_weight": 1.0, "task_reward_coef": 0.0}},
        "actor_rollout_ref": {"actor": {"ppo_epochs": ppo_epochs, "use_kl_loss": use_kl_loss,
                                        "entropy_coeff": entropy_coeff,
                                        "ppo_mini_batch_size": ppo_mini_batch_size},
                              "rollout": {"n": rollout_n}},
    })


def make_trainer(trainer_source, config, *, use_teacher_policy=False, bg_loop=None):
    trainer = SimpleNamespace(
        config=config, use_teacher_policy=use_teacher_policy, bg_loop=bg_loop,
        use_agent_teacher=bool((config.algorithm.get("teacher") or {}).get("endpoints_file")),
        _warned_opd_optimizer_steps=False,
    )
    for name, fn in trainer_source["functions"].items():
        setattr(trainer, name, types.MethodType(fn, trainer))
    return trainer


class Batch:
    """Minimal DataProto stand-in: tensor fields in .batch, len() = rows."""

    def __init__(self, rows=3, width=5):
        self.batch = {"input_ids": torch.arange(rows * width).reshape(rows, width),
                      "attention_mask": torch.ones(rows, width, dtype=torch.long),
                      "action_mask": torch.ones(rows, width)}

    def __len__(self):
        return self.batch["input_ids"].shape[0]


# ---- teacher step --------------------------------------------------------------------------------

def test_teacher_step_runs_on_the_background_loop_and_fills_the_batch(trainer_source, background_loop,
                                                                      monkeypatch):
    calls = {}

    def fake_teacher(snapshot, teacher_config):
        calls["thread"] = threading.current_thread()
        calls["config"] = teacher_config
        time.sleep(0.3)                                   # scoring takes a while
        return torch.full(snapshot.batch["input_ids"].shape, -0.5), {"teacher/tokens": 15.0}

    monkeypatch.setattr(agent_algorithms, "compute_teacher_log_probs", fake_teacher)
    trainer = make_trainer(trainer_source, make_config(), bg_loop=background_loop)
    batch, metrics = Batch(), {}

    started = time.monotonic()
    pending = trainer._start_agent_teacher(batch)
    assert time.monotonic() - started < 0.1              # returns at once: scoring overlaps later steps
    batch.batch["old_log_probs"] = torch.zeros(3, 5)      # later steps add fields meanwhile
    trainer._finish_agent_teacher(pending, batch, metrics)

    torch.testing.assert_close(batch.batch["teacher_log_probs"], torch.full((3, 5), -0.5))
    assert metrics == {"teacher/tokens": 15.0}
    assert calls["thread"] is not threading.main_thread()
    assert calls["config"].endpoints_file == "/tmp/teacher.json"


def test_teacher_errors_surface_when_the_step_is_joined(trainer_source, background_loop, monkeypatch):
    def failing(snapshot, teacher_config):
        raise agent_algorithms.TeacherUnavailableError("teacher down for 1800s")

    monkeypatch.setattr(agent_algorithms, "compute_teacher_log_probs", failing)
    trainer = make_trainer(trainer_source, make_config(), bg_loop=background_loop)
    batch = Batch()
    pending = trainer._start_agent_teacher(batch)
    with pytest.raises(agent_algorithms.TeacherUnavailableError):
        trainer._finish_agent_teacher(pending, batch, {})
    assert "teacher_log_probs" not in batch.batch


def test_rows_changed_during_scoring_are_rejected(trainer_source, background_loop, monkeypatch):
    monkeypatch.setattr(agent_algorithms, "compute_teacher_log_probs",
                        lambda snap, cfg: (torch.zeros(snap.batch["input_ids"].shape), {}))
    trainer = make_trainer(trainer_source, make_config(), bg_loop=background_loop)
    batch = Batch()
    pending = trainer._start_agent_teacher(batch)
    batch.batch["input_ids"] = batch.batch["input_ids"].flip(0)   # e.g. a reordering step
    with pytest.raises(RuntimeError, match="misalign"):
        trainer._finish_agent_teacher(pending, batch, {})


def test_disabled_teacher_does_nothing(trainer_source, monkeypatch):
    monkeypatch.setattr(agent_algorithms, "compute_teacher_log_probs",
                        lambda *a: pytest.fail("teacher must not be called"))
    trainer = make_trainer(trainer_source, make_config(adv_estimator="grpo", endpoints_file=None))
    batch, metrics = Batch(), {}
    assert trainer._start_agent_teacher(batch) is None
    trainer._finish_agent_teacher(None, batch, metrics)
    assert "teacher_log_probs" not in batch.batch and metrics == {}


# ---- startup checks ----------------------------------------------------------------------------

def test_opd_requires_the_teacher(trainer_source):
    trainer = make_trainer(trainer_source, make_config(endpoints_file=None))
    with pytest.raises(ValueError, match="needs algorithm.teacher.endpoints_file"):
        trainer._check_agent_teacher_config()


def test_teacher_and_verl_distillation_are_exclusive(trainer_source):
    trainer = make_trainer(trainer_source, make_config(), use_teacher_policy=True)
    with pytest.raises(ValueError, match="enable only one"):
        trainer._check_agent_teacher_config()


def test_opd_warns_on_settings_that_change_the_objective(trainer_source, capsys):
    make_trainer(trainer_source, make_config())._check_agent_teacher_config()
    assert capsys.readouterr().out == ""
    make_trainer(trainer_source, make_config(ppo_epochs=2, use_kl_loss=True,
                                             entropy_coeff=1e-3))._check_agent_teacher_config()
    out = capsys.readouterr().out
    assert "ppo_epochs=2" in out and "use_kl_loss=True" in out and "entropy_coeff=0.001" in out


def test_other_estimators_are_unaffected(trainer_source, capsys):
    make_trainer(trainer_source, make_config(adv_estimator="grpo", endpoints_file=None,
                                             ppo_epochs=4))._check_agent_teacher_config()
    assert capsys.readouterr().out == ""


# ---- metrics ------------------------------------------------------------------------------------

def test_opd_metrics_and_optimizer_step_warning(trainer_source, monkeypatch, capsys):
    monkeypatch.setattr(agent_algorithms, "compute_opd_metrics", lambda batch, cfg: {"opd/kl_k1_mean": 0.2})
    trainer = make_trainer(trainer_source, make_config(ppo_mini_batch_size=4))

    assert trainer._agent_algorithm_metrics(Batch(rows=4)) == {"opd/kl_k1_mean": 0.2, "opd/optimizer_steps": 1.0}
    assert capsys.readouterr().out == ""
    metrics = trainer._agent_algorithm_metrics(Batch(rows=6))          # 6 rows / 4 per update -> 2 steps
    assert metrics["opd/optimizer_steps"] == 2.0
    assert "2 optimizer steps per batch" in capsys.readouterr().out
    trainer._agent_algorithm_metrics(Batch(rows=6))
    assert capsys.readouterr().out == ""                                # warned once


def test_non_opd_estimators_log_no_opd_metrics(trainer_source, monkeypatch):
    monkeypatch.setattr(agent_algorithms, "compute_opd_metrics", lambda *a: pytest.fail("not for grpo"))
    trainer = make_trainer(trainer_source, make_config(adv_estimator="grpo", endpoints_file=None))
    assert trainer._agent_algorithm_metrics(Batch()) == {}


# ---- where fit() calls the hook ----------------------------------------------------------------

def test_fit_starts_after_the_mask_shift_and_joins_before_the_advantage(trainer_source):
    source = ast.get_source_segment(TRAINER.read_text(), trainer_source["fit"])
    order = [source.index(marker) for marker in (
        "batch.batch['action_mask'] = action_mask_aligned",   # position convention is now in place
        "self._start_agent_teacher(batch)",
        'with marked_timer("old_log_prob"',                    # overlaps old_log_prob / ref
        "self._finish_agent_teacher(pending_agent_teacher, batch, metrics)",
        "batch = compute_advantage(",
        "compute_data_metrics(batch=batch",
        "self._agent_algorithm_metrics(batch)",
    )]
    assert order == sorted(order)
