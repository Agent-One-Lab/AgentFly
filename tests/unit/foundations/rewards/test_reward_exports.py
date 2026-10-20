"""Lazy reward exports must resolve to rewards, never to same-named submodules.

Importing a submodule binds it onto the package, so an exported name that
matches a file in ``agentfly/rewards/`` would shadow the reward once the
built-ins are loaded, and ``from agentfly.rewards import <name>`` would return
the module.
"""
import pkgutil
import types

import agentfly.rewards as rewards
from agentfly.rewards.reward_base import ensure_builtins_loaded


def test_no_export_shares_a_submodule_name():
    submodules = {m.name for m in pkgutil.iter_modules(rewards.__path__)}
    assert set(rewards.__all__) & submodules == set()


def test_exports_are_not_modules_after_builtins_load():
    ensure_builtins_loaded()
    for name in rewards.__all__:
        assert not isinstance(getattr(rewards, name), types.ModuleType), name
