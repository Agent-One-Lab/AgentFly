"""Unit tests for the typed reward result class and its from_raw factory."""
import pytest

from agentfly.rewards.types import RewardResult, RewardReturn


class TestRewardResultFromRaw:
    def test_from_raw_float(self):
        r = RewardResult.from_raw(0.7)
        assert r.reward == 0.7
        assert r.extras == {}

    def test_from_raw_int_coerced_to_float(self):
        r = RewardResult.from_raw(1)
        assert r.reward == 1.0
        assert isinstance(r.reward, float)
        assert r.extras == {}

    def test_from_raw_bool_coerced_to_float(self):
        # We allow bool because it's int-like; verify it lands as a clean float.
        r = RewardResult.from_raw(True)
        assert r.reward == 1.0

    def test_from_raw_dict_with_extras(self):
        r = RewardResult.from_raw({"reward": 0.5, "f1": 0.8, "em": 1.0})
        assert r.reward == 0.5
        assert r.extras == {"f1": 0.8, "em": 1.0}

    def test_from_raw_dict_only_reward(self):
        r = RewardResult.from_raw({"reward": 0.3})
        assert r.reward == 0.3
        assert r.extras == {}

    def test_from_raw_none_yields_zero(self):
        r = RewardResult.from_raw(None)
        assert r.reward == 0.0
        assert r.extras == {}

    def test_from_raw_passthrough(self):
        original = RewardResult(reward=0.5, extras={"x": 1})
        assert RewardResult.from_raw(original) is original

    def test_from_raw_dict_missing_reward_key(self):
        with pytest.raises(ValueError, match="reward key"):
            RewardResult.from_raw({"score": 0.5})

    def test_from_raw_unsupported_type(self):
        with pytest.raises(ValueError):
            RewardResult.from_raw("not a number")


class TestRewardReturn:
    def test_typed_dict_is_annotation_only(self):
        # RewardReturn is a TypedDict; runtime objects are plain dicts.
        x: RewardReturn = {"reward": 0.5, "f1": 0.8}
        assert isinstance(x, dict)
