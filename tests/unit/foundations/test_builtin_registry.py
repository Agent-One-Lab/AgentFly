"""Every built-in tool and reward stays importable and registered by name.

The built-ins register themselves as a side effect of the imports in ``tools/impls.py`` and
``rewards/impls.py``. Those imports look unused to a linter, so this guards against a tool such
as ``ruff --fix`` deleting them (their ``__all__`` lists are what keep ruff from doing so).
"""

import pytest

import agentfly.rewards
import agentfly.tools
from agentfly.rewards import impls as reward_impls
from agentfly.rewards.reward_base import _CallableRewardClass, list_available_rewards
from agentfly.tools import impls as tool_impls
from agentfly.tools.decorator import _CallableToolClass
from agentfly.tools.registry.registry import list_available_tools


@pytest.mark.parametrize(
    ("package", "impls"),
    [(agentfly.tools, tool_impls), (agentfly.rewards, reward_impls)],
    ids=["tools", "rewards"],
)
def test_builtins_are_exported(package, impls):
    for name in impls.__all__:
        assert getattr(impls, name) is getattr(package, name), name


def test_builtin_tools_are_registered():
    registered = set(list_available_tools())
    tools = [getattr(tool_impls, n) for n in tool_impls.__all__]
    tools = [t for t in tools if isinstance(t, _CallableToolClass)]  # made by @tool; skips classes and helpers
    assert len(tools) >= 25
    missing = sorted(t.name for t in tools if t.name not in registered)
    assert not missing, f"built-in tools not registered: {missing}"


def test_builtin_rewards_are_registered():
    registered = set(list_available_rewards())
    rewards = [getattr(reward_impls, n) for n in reward_impls.__all__]
    rewards = [r for r in rewards if isinstance(r, _CallableRewardClass)]  # made by @reward
    assert len(rewards) == len(reward_impls.__all__)
    missing = sorted(r.name for r in rewards if r.name.lower() not in registered)
    assert not missing, f"built-in rewards not registered: {missing}"
