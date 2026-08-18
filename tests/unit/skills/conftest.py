"""Shared fixtures for skill tool tests.

Each test gets a fresh ``Context`` pointing at ``skills/test-skills`` so the
per-rollout ``_loaded_skills`` state doesn't leak across tests.
"""

import json
from pathlib import Path

import pytest

from agentfly.core import Context


REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_SKILLS_ROOT = REPO_ROOT / "skills" / "test-skills"


@pytest.fixture(scope="session")
def test_skills_root() -> Path:
    if not TEST_SKILLS_ROOT.is_dir():
        pytest.skip(f"test skills not found at {TEST_SKILLS_ROOT}")
    return TEST_SKILLS_ROOT


@pytest.fixture
def context(test_skills_root, request) -> Context:
    return Context(
        rollout_id=request.node.nodeid,
        metadata={"skills_root": str(test_skills_root)},
    )


def _parse_observation(result):
    """Tools wrap output in ``{"observation": "...json..."}``; pull the JSON out."""
    obs = result["observation"] if isinstance(result, dict) else result
    return json.loads(obs)


@pytest.fixture
def parse_observation():
    return _parse_observation
