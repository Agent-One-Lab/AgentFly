"""Reward metrics on training batches are marked for verl to log as ``reward_extra/<key>``."""

import numpy as np
import torch

from agentfly.agents.rollout.conversion import mark_reward_extras


def test_marks_numeric_reward_metric_columns():
    inputs = {
        "rm_scores": torch.zeros(3, 4),                     # the reward tensor itself, not an extra
        "rm_em": np.array([1.0, 0.0, 1.0]),
        "rm_guesses": np.array([3, 5, 7], dtype=object),    # step rollouts store metrics as object
        "rm_output": np.array(["a", "b", "c"], dtype=object),  # text: not loggable
        "uid": np.array(["x", "y", "z"], dtype=object),
    }
    assert sorted(mark_reward_extras(inputs)) == ["rm_em", "rm_guesses"]
    assert inputs["rm_guesses"].dtype == np.float64        # converted so verl's numeric check passes
    assert inputs["rm_output"].dtype == object             # left as is
