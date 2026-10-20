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



USER_ESTIMATOR = """
from agentfly.algorithms import register_estimator


@register_estimator("my_grpo", layout="per_segment")
def my_grpo(data, config):
    return None, None


@register_estimator("my_grpo", layout="per_step")
def my_grpo_steps(data, config):
    return None, None


def not_an_estimator(data, config):
    return None, None
"""


@pytest.fixture
def user_estimator_file(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "ESTIMATOR_REGISTRY", dict(registry.ESTIMATOR_REGISTRY))
    path = tmp_path / "my_algos.py"
    path.write_text(USER_ESTIMATOR)
    return path


def test_resolve_estimator_keeps_bare_names():
    assert algorithms.resolve_estimator("gigpo") == "gigpo"
    assert algorithms.resolve_estimator("gae") == "gae"  # verl built-in, left for verl


def test_resolve_estimator_imports_file_reference(user_estimator_file):
    assert not algorithms.has_estimator("my_grpo")
    assert algorithms.resolve_estimator(f"{user_estimator_file}:my_grpo") == "my_grpo"
    # Importing the module registers every layout under the name, not just the referenced fn.
    assert algorithms.estimator_layouts("my_grpo") == ["per_segment", "per_step"]


def test_resolve_estimator_imports_module_reference(user_estimator_file, monkeypatch):
    monkeypatch.syspath_prepend(str(user_estimator_file.parent))
    monkeypatch.delitem(__import__("sys").modules, "my_algos", raising=False)
    assert algorithms.resolve_estimator("my_algos:my_grpo_steps") == "my_grpo"


def test_resolve_estimator_rejects_unregistered_function(user_estimator_file):
    with pytest.raises(ValueError, match="not point to a function registered"):
        algorithms.resolve_estimator(f"{user_estimator_file}:not_an_estimator")


ESTIMATOR_WITH_METRIC = """
from agentfly.algorithms import register_estimator

CALLS = 0


@register_estimator("shared_grpo", layout="per_segment")
def shared_grpo(data, config):
    global CALLS
    CALLS += 1
    return None, None


def calls_metric(batch, config):
    return {"calls": CALLS}
"""


@pytest.fixture
def isolated_registries(monkeypatch):
    monkeypatch.setattr(registry, "ESTIMATOR_REGISTRY", dict(registry.ESTIMATOR_REGISTRY))
    monkeypatch.setattr(registry, "METRICS_REGISTRY", {k: list(v) for k, v in registry.METRICS_REGISTRY.items()})


def test_file_reference_executes_the_file_once(tmp_path, isolated_registries):
    from agentfly.utils.references import load_object

    path = tmp_path / "algos.py"
    path.write_text(ESTIMATOR_WITH_METRIC)
    ref = f"{path}:shared_grpo"
    assert algorithms.resolve_estimator(ref) == "shared_grpo"
    assert algorithms.resolve_estimator(ref) == "shared_grpo"  # resolving again is fine
    assert load_object(ref) is load_object(f"{tmp_path}/./algos.py:shared_grpo")


def test_estimator_and_metric_from_one_file_share_the_module(tmp_path, isolated_registries):
    path = tmp_path / "algos.py"
    path.write_text(ESTIMATOR_WITH_METRIC)
    name = algorithms.resolve_estimator(f"{path}:shared_grpo")
    (metric,) = algorithms.load_metrics([f"{path}:calls_metric"])  # used to raise "already registered"
    algorithms.get_estimator(name, "per_segment")(None, None)
    assert metric(None, None) == {"calls": 1}  # the metric sees the estimator's module state


def test_files_with_the_same_name_stay_separate(tmp_path, isolated_registries):
    from agentfly.utils.references import load_object

    for sub, value in (("a", 1), ("b", 2)):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "tools.py").write_text(f"VALUE = {value}\n")
    assert load_object(f"{tmp_path}/a/tools.py:VALUE") == 1
    assert load_object(f"{tmp_path}/b/tools.py:VALUE") == 2


def test_file_reference_reuses_an_imported_module(tmp_path, monkeypatch, isolated_registries):
    import importlib

    (tmp_path / "imported_algos.py").write_text(ESTIMATOR_WITH_METRIC)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(__import__("sys").modules, "imported_algos", raising=False)
    module = importlib.import_module("imported_algos")
    assert algorithms.resolve_estimator(f"{tmp_path}/imported_algos.py:shared_grpo") == "shared_grpo"
    assert algorithms.get_estimator("shared_grpo", "per_segment") is module.shared_grpo


def test_failed_file_load_can_be_retried(tmp_path):
    from agentfly.utils.references import load_object

    path = tmp_path / "broken.py"
    path.write_text("raise RuntimeError('not yet')\n")
    with pytest.raises(RuntimeError, match="not yet"):
        load_object(f"{path}:VALUE")
    path.write_text("VALUE = 3\n")
    assert load_object(f"{path}:VALUE") == 3


def test_reregistering_the_same_definition_replaces_it(isolated_registries):
    def define():  # e.g. re-running a notebook cell: same module and qualname, new object
        @registry.register_estimator("redefined", layout="per_step")
        def redefined(data, config):
            return None, None

        @registry.register_metrics("redefined")
        def redefined_metric(batch, config):
            return {}

        return redefined

    define()
    latest = define()
    assert algorithms.get_estimator("redefined", "per_step") is latest
    assert len(registry.METRICS_REGISTRY["redefined"]) == 1


def test_same_name_from_two_files_raises_with_both_locations(tmp_path, isolated_registries):
    from agentfly.utils.references import load_object

    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "algos.py").write_text(ESTIMATOR_WITH_METRIC)
    load_object(f"{tmp_path}/a/algos.py:shared_grpo")
    with pytest.raises(ValueError, match=r"already registered by _agentfly_ref_algos_\w+:shared_grpo; "
                                         r"_agentfly_ref_algos_\w+:shared_grpo registers the same name"):
        load_object(f"{tmp_path}/b/algos.py:shared_grpo")

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
