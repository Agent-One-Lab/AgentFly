"""Validation aggregation must tolerate reward extras missing on some rows (None)."""
import math

from agentfly.verl.trainer.ppo.metric_utils import process_validation_metrics


def test_none_reward_extras_are_skipped_not_averaged():
    data_sources = ["ds", "ds", "ds", "ds"]
    uids = ["u1", "u1", "u2", "u2"]
    infos = {
        "reward": [1.0, 0.0, 0.0, 0.0],
        "rm_exit": [0, None, None, 1],          # key missing on two trajectories
        "rm_error": [None, "boom", "boom", ""],  # string column, first entry None
        "rm_all_none": [None, None, None, None],
    }
    out = process_validation_metrics(data_sources, uids, infos)
    assert math.isclose(out["ds"]["reward"]["mean@2"], 0.25)
    assert math.isclose(out["ds"]["rm_exit"]["mean@1"], 0.5)  # (0 + 1) / 2 over the surviving values
    assert "rm_error" not in out["ds"]
    assert "rm_all_none" not in out["ds"]
