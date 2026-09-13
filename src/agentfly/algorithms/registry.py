"""Registry of agent advantage estimators, keyed by estimator name and batch layout.

An agent training batch declares how its rows relate to the environment in
``meta_info["layout"]``:

- ``"per_step"`` — one row per environment step (``StepRollout``);
- ``"per_segment"`` — one row per conversation segment, several turns per row
  (``ChainRollout``; context folding can split one trajectory into several segments).

An algorithm registers one function per layout it supports::

    @register_estimator("gigpo", layout="per_step")
    def gigpo_per_step(data, config):
        ...
        return advantages, returns

Estimators take the whole ``DataProto`` (they read whichever batch fields their layout
carries) and the trainer's algorithm config, and return ``(advantages, returns)`` tensors of
shape ``[B, L]``. The trainer looks estimators up here first and falls back to verl's own
estimators for names that are not registered.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

LAYOUTS = ("per_step", "per_segment")

ESTIMATOR_REGISTRY: Dict[Tuple[str, str], Callable] = {}


def estimator_name(name_or_enum: Any) -> str:
    """The plain string name of an estimator given a string or an ``Enum`` member."""
    return name_or_enum.value if isinstance(name_or_enum, Enum) else str(name_or_enum)


def register_estimator(name: str, *, layout: str) -> Callable[[Callable], Callable]:
    """Register ``fn(data, config) -> (advantages, returns)`` for ``name`` on ``layout``."""
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown batch layout {layout!r}; expected one of {LAYOUTS}.")

    def decorator(fn: Callable) -> Callable:
        key = (name, layout)
        if key in ESTIMATOR_REGISTRY and ESTIMATOR_REGISTRY[key] is not fn:
            raise ValueError(
                f"Estimator {name!r} is already registered for layout {layout!r}: "
                f"{ESTIMATOR_REGISTRY[key]!r}"
            )
        ESTIMATOR_REGISTRY[key] = fn
        return fn

    return decorator


def estimator_layouts(name_or_enum: Any) -> List[str]:
    """Layouts ``name`` is registered for (empty when AgentFly does not provide it)."""
    name = estimator_name(name_or_enum)
    return sorted(layout for (n, layout) in ESTIMATOR_REGISTRY if n == name)


def has_estimator(name_or_enum: Any) -> bool:
    """True if AgentFly provides an estimator under this name (for any layout)."""
    return bool(estimator_layouts(name_or_enum))


def get_estimator(name_or_enum: Any, layout: Any) -> Callable:
    """The estimator for ``name`` on ``layout``; raises with the supported layouts if absent."""
    name = estimator_name(name_or_enum)
    fn = ESTIMATOR_REGISTRY.get((name, layout))
    if fn is None:
        supported = estimator_layouts(name)
        if not supported:
            raise KeyError(f"No agent estimator named {name!r}.")
        raise KeyError(
            f"Estimator {name!r} does not support batch layout {layout!r} "
            f"(meta_info['layout']); supported layouts: {supported}."
        )
    return fn
