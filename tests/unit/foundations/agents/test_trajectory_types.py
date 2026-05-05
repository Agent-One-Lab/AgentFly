"""Unit tests for the typed Trajectory and RunResult classes."""
import pytest

from agentfly.agents.types import RunResult, Trajectory


class TestTrajectoryConstruction:
    def test_minimal(self):
        t = Trajectory(segments=[[{"role": "user", "content": "hi"}]])
        assert t.reward is None
        assert t.segment_rewards is None
        assert t.metrics == {}
        assert t.metadata == {}
        assert t.runtime_info == {}
        assert t.finish_reason is None

    def test_full(self):
        t = Trajectory(
            segments=[[{"role": "user", "content": "hi"}]],
            reward=0.7,
            segment_rewards=[0.7],
            metrics={"f1": 0.8, "em": 1.0},
            finish_reason="terminal",
            rollout_time_sec=1.5,
            chain_id="cid-1",
            group_id="gid-1",
            chain_idx=0,
            group_idx=0,
            metadata={"answer": "42", "task_id": 7},
            runtime_info={"k": "v"},
        )
        assert t.reward == 0.7
        assert t.metrics == {"f1": 0.8, "em": 1.0}
        assert t.metadata == {"answer": "42", "task_id": 7}
        assert t.chain_id == "cid-1"
        assert t.finish_reason == "terminal"


class TestTrajectoryDerivedProperties:
    def test_single_segment_is_not_segmented(self):
        t = Trajectory(segments=[[{"role": "user", "content": "hi"}]])
        assert t.is_segmented is False
        assert t.num_segments == 1

    def test_multiple_segments_is_segmented(self):
        t = Trajectory(
            segments=[
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ]
        )
        assert t.is_segmented is True
        assert t.num_segments == 2


class TestTrajectorySerialization:
    def test_json_roundtrip(self):
        t = Trajectory(
            segments=[[{"role": "user", "content": "hi"}]],
            reward=0.5,
            metrics={"f1": 0.8},
            metadata={"answer": "42"},
        )
        t2 = Trajectory.model_validate_json(t.model_dump_json())
        assert t2.reward == t.reward
        assert t2.metrics == t.metrics
        assert t2.metadata == t.metadata
        assert t2.segments == t.segments

    def test_model_dump_exclude_segments(self):
        t = Trajectory(
            segments=[[{"role": "user", "content": "hi"}]],
            reward=0.5,
            metrics={"f1": 0.8},
            chain_id="cid-1",
        )
        info = t.model_dump(exclude={"segments"})
        assert "segments" not in info
        assert info["reward"] == 0.5
        assert info["metrics"] == {"f1": 0.8}
        assert info["chain_id"] == "cid-1"


class TestRunResult:
    def test_empty(self):
        result = RunResult(trajectories=[])
        assert len(result) == 0
        assert result.rewards == []
        assert result.reward_extras == {}

    def test_rewards_list_in_order(self):
        ts = [
            Trajectory(segments=[[{"role": "user", "content": "x"}]], reward=1.0),
            Trajectory(segments=[[{"role": "user", "content": "x"}]], reward=0.5),
            Trajectory(segments=[[{"role": "user", "content": "x"}]], reward=None),
        ]
        result = RunResult(trajectories=ts)
        assert result.rewards == [1.0, 0.5, None]

    def test_reward_extras_alignment(self):
        # First trajectory has both metrics; second has only one; third has none.
        ts = [
            Trajectory(
                segments=[[{"role": "user", "content": "x"}]],
                reward=1.0,
                metrics={"f1": 0.8, "em": 1.0},
            ),
            Trajectory(
                segments=[[{"role": "user", "content": "x"}]],
                reward=0.5,
                metrics={"f1": 0.5},
            ),
            Trajectory(
                segments=[[{"role": "user", "content": "x"}]],
                reward=0.0,
            ),
        ]
        result = RunResult(trajectories=ts)
        extras = result.reward_extras
        # Each list should have exactly len(trajectories) entries; missing → None.
        assert extras == {
            "em": [1.0, None, None],
            "f1": [0.8, 0.5, None],
        }

    def test_iteration(self):
        ts = [
            Trajectory(segments=[[{"role": "user", "content": "x"}]], reward=float(i))
            for i in range(3)
        ]
        result = RunResult(trajectories=ts)
        for i, t in enumerate(result):
            assert t.reward == float(i)

    def test_indexing(self):
        ts = [
            Trajectory(segments=[[{"role": "user", "content": "x"}]], reward=float(i))
            for i in range(3)
        ]
        result = RunResult(trajectories=ts)
        assert result[0].reward == 0.0
        assert result[2].reward == 2.0

    def test_json_roundtrip_preserves_extras_alignment(self):
        ts = [
            Trajectory(
                segments=[[{"role": "user", "content": "x"}]],
                reward=1.0,
                metrics={"f1": 0.8},
            ),
            Trajectory(
                segments=[[{"role": "user", "content": "x"}]],
                reward=0.5,
            ),
        ]
        result = RunResult(trajectories=ts)
        result2 = RunResult.model_validate_json(result.model_dump_json())
        assert result2.rewards == result.rewards
        assert result2.reward_extras == result.reward_extras


class TestTrajectoryReservedKeysLogic:
    """Verify the chain-side mapping invariants assumed by ChainRollout.get_trajectories.

    These don't import the chain (which has heavy deps) — they just confirm
    that the field names we map into match the Trajectory schema, so the
    chain-side code in chain_base.py won't silently drift.
    """

    def test_typed_field_names_match_chain_info_keys(self):
        # The chain stores these keys on chain.info; get_trajectories maps them
        # to typed fields. This test fails if a field is renamed without
        # updating the chain code.
        expected_typed_fields = {
            "reward",
            "finish_reason",
            "rollout_time_sec",
            "chain_id",
            "group_id",
            "chain_idx",
            "group_idx",
        }
        traj_fields = set(Trajectory.model_fields.keys())
        missing = expected_typed_fields - traj_fields
        assert not missing, f"Trajectory is missing fields: {missing}"
