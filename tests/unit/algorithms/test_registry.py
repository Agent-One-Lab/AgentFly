"""Estimator registry and the trainer's lookup by estimator name + batch layout."""

from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import agentfly.algorithms as algorithms
from agentfly.algorithms import registry
from agentfly.algorithms.gigpo import gigpo_per_segment, gigpo_per_step


def test_gigpo_registered_for_both_layouts():
    assert algorithms.estimator_layouts("gigpo") == ["per_segment", "per_step"]
    assert algorithms.get_estimator("gigpo", "per_step") is gigpo_per_step
    assert algorithms.get_estimator("gigpo", "per_segment") is gigpo_per_segment


def test_enum_names_resolve():
    class Est(str, Enum):
        GIGPO = "gigpo"

    assert algorithms.has_estimator(Est.GIGPO)
    assert algorithms.get_estimator(Est.GIGPO, "per_step") is gigpo_per_step


def test_names_not_provided_fall_through_to_verl():
    assert not algorithms.has_estimator("grpo")
    assert not algorithms.has_estimator("gae")


def test_missing_or_unknown_layout_names_supported_layouts():
    with pytest.raises(KeyError, match="supported layouts: \\['per_segment', 'per_step'\\]"):
        algorithms.get_estimator("gigpo", None)
    with pytest.raises(KeyError, match="No agent estimator named 'nope'"):
        algorithms.get_estimator("nope", "per_step")


def test_register_rejects_unknown_layout_and_duplicates(monkeypatch):
    monkeypatch.setattr(registry, "ESTIMATOR_REGISTRY", {})
    with pytest.raises(ValueError, match="Unknown batch layout"):
        registry.register_estimator("x", layout="per_token")

    @registry.register_estimator("x", layout="per_step")
    def first(data, config):
        return None

    with pytest.raises(ValueError, match="already registered"):
        @registry.register_estimator("x", layout="per_step")
        def second(data, config):
            return None


class FakeData:
    def __init__(self, batch, non_tensor, layout):
        self.batch = batch
        self.non_tensor_batch = non_tensor
        self.meta_info = {"layout": layout}


def test_missing_batch_field_names_the_field_and_layout():
    data = FakeData({"token_level_rewards": torch.zeros(1, 1), "action_mask": torch.ones(1, 1)},
                    {"uid": np.array(["g"], dtype=object)}, "per_segment")
    with pytest.raises(KeyError, match="gigpo on a 'per_segment' batch needs the field 'step_observations'"):
        gigpo_per_segment(data, None)


def test_estimator_reads_gigpo_settings_from_config():
    """``gigpo_mode`` / ``gamma`` / ``step_advantage_w`` come from the algorithm config."""
    tlr = torch.zeros(4, 1)
    tlr[:, 0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    obj = lambda v: np.array(v + [None], dtype=object)[:-1]  # noqa: E731 — 1-D object array of lists
    data = FakeData(
        {"token_level_rewards": tlr, "action_mask": torch.ones(4, 1)},
        {"uid": np.array(["g"] * 4, dtype=object), "step_observations": obj([["A"]] * 4),
         "step_rewards": obj([[0.0]] * 4)},
        "per_segment",
    )
    mean_only, _ = gigpo_per_segment(data, SimpleNamespace(gigpo_mode="mean_norm", gamma=0.95, step_advantage_w=1.0,
                                                           invalid_action_penalty_coef=0.1))
    with_std, _ = gigpo_per_segment(data, {"gigpo_mode": "mean_std_norm"})
    assert np.allclose(mean_only[:, 0].numpy(), [7.5, -2.5, -2.5, -2.5], atol=1e-4)
    assert np.allclose(with_std[:, 0].numpy(), [1.5, -0.5, -0.5, -0.5], atol=1e-4)


def test_trainer_compute_advantage_uses_agent_estimator():
    ray_trainer = pytest.importorskip("agentfly.verl.trainer.ppo.ray_trainer")
    from agentfly.verl.protocol import DataProto

    tlr = torch.zeros(4, 1)
    tlr[:, 0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    obj = lambda v: np.array(v + [None], dtype=object)[:-1]  # noqa: E731
    data = DataProto.from_single_dict(
        {"token_level_rewards": tlr, "action_mask": torch.ones(4, 1),
         "uid": np.array(["g"] * 4, dtype=object), "step_observations": obj([["A"]] * 4),
         "step_rewards": obj([[0.0]] * 4)},
        meta_info={"use_agent": True, "layout": "per_segment"},
    )
    out = ray_trainer.compute_advantage(data, adv_estimator="gigpo", config={"gigpo_mode": "mean_norm"})
    assert np.allclose(out.batch["advantages"][:, 0].numpy(), [7.5, -2.5, -2.5, -2.5], atol=1e-4)
    assert torch.equal(out.batch["advantages"], out.batch["returns"])
