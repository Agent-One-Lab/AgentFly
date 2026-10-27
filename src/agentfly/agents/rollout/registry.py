"""Rollout strategy resolution.

``resolve_rollout`` turns a ``run(..., rollout=...)`` argument into a concrete
:class:`~agentfly.agents.rollout.base.Rollout` instance:

- a ``Rollout`` instance passes through unchanged (lets callers pass a configured
  strategy, e.g. ``StepRollout(history_length=2)``);
- a string is a registered name (``"chain"``) OR an import reference
  (``"pkg.module:MyRollout"`` / ``"/path.py:MyRollout"``) resolved via
  :func:`agentfly.utils.references.resolve_reference` — the same name-or-path idiom
  used for tools/rewards/agents — and then default-constructed.
"""

from ...utils.references import resolve_reference
from .base import Rollout
from .strategies.chain_rollout import ChainRollout
from .strategies.step_rollout import StepRollout

ROLLOUT_REGISTRY = {
    "chain": ChainRollout,
    "step": StepRollout,
}


def register_rollout(name: str, rollout_cls) -> None:
    """Register a rollout strategy class under ``name`` (case-insensitive)."""
    ROLLOUT_REGISTRY[name.lower()] = rollout_cls


def resolve_rollout(spec, **kwargs) -> Rollout:
    """Resolve ``spec`` to a ``Rollout`` instance, constructing it with ``**kwargs``.

    ``kwargs`` are the rollout's constructor args (e.g. ``prompt_builder``,
    ``history_length`` for :class:`StepRollout`), so a config can select *and configure*
    a strategy: ``resolve_rollout("step", prompt_builder="alfworld_flat", history_length=2)``.
    An already-constructed ``Rollout`` instance passes through unchanged (kwargs ignored).
    """
    if isinstance(spec, Rollout):
        return spec
    rollout_cls = resolve_reference(spec, ROLLOUT_REGISTRY, "rollout")
    return rollout_cls(**kwargs)
