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


def _definition(fn: Callable) -> str:
    """Where ``fn`` is defined: its module and qualified name (the identity pickle uses)."""
    return f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', repr(fn))}"


def register_estimator(name: str, *, layout: str) -> Callable[[Callable], Callable]:
    """Register ``fn(data, config) -> (advantages, returns)`` for ``name`` on ``layout``.

    Registering the same definition again (re-running its module or notebook cell) replaces
    the entry; a different definition under a taken name raises.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown batch layout {layout!r}; expected one of {LAYOUTS}.")

    def decorator(fn: Callable) -> Callable:
        key = (name, layout)
        existing = ESTIMATOR_REGISTRY.get(key)
        if existing is not None and _definition(existing) != _definition(fn):
            raise ValueError(
                f"Estimator {name!r} ({layout}) is already registered by {_definition(existing)}; "
                f"{_definition(fn)} registers the same name."
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


def resolve_estimator(name_or_ref: Any) -> str:
    """The estimator name for ``algorithm.adv_estimator``, given a name or an import reference.

    A bare name (``"gigpo"``, or a verl built-in such as ``"gae"``) is returned unchanged. A
    reference (``"pkg.mod:fn"`` or ``"/path/file.py:fn"``) is imported, which fires the module's
    ``@register_estimator`` decorators, and the name ``fn`` was registered under is returned, so
    every layout registered under that name is available.
    """
    name = estimator_name(name_or_ref)
    from ..utils.references import is_reference, load_object

    if has_estimator(name) or not is_reference(name):
        return name
    fn = load_object(name)
    names = {n for (n, _), f in ESTIMATOR_REGISTRY.items() if f is fn}
    if len(names) != 1:
        raise ValueError(
            f"Estimator reference {name!r} does not point to a function registered with "
            f"@register_estimator (found registered names: {sorted(names)})."
        )
    return names.pop()


# ------------------------------------------------------------------------------------ #
# Metrics: per-estimator diagnostics logged after the advantage step                    #
# ------------------------------------------------------------------------------------ #
METRICS_REGISTRY: Dict[str, List[Callable]] = {}


def register_metrics(*names: str) -> Callable[[Callable], Callable]:
    """Register ``fn(batch, config) -> dict`` to run for each estimator in ``names``.

    The trainer calls :func:`compute_estimator_metrics` on the batch after the advantage step
    (it still carries ``advantages``/``returns``/``values`` and the rollout's fields).
    """

    def decorator(fn: Callable) -> Callable:
        for name in names:
            fns = METRICS_REGISTRY.setdefault(name, [])
            # Registering the same definition again replaces it instead of adding a copy.
            fns[:] = [f for f in fns if _definition(f) != _definition(fn)] + [fn]
        return fn

    return decorator


def compute_estimator_metrics(name_or_enum: Any, batch: Any, config: Any) -> Dict[str, float]:
    """All registered metrics for this estimator (empty when none are registered)."""
    metrics: Dict[str, float] = {}
    for fn in METRICS_REGISTRY.get(estimator_name(name_or_enum), []):
        metrics.update(fn(batch, config))
    return metrics


def load_metrics(refs: Any) -> List[Callable]:
    """Metric functions for ``algorithm.metrics``: each import reference (``"pkg.mod:fn"`` or
    ``"/path/file.py:fn"``) is imported and must name a callable ``fn(batch, config) -> dict``.
    They run every step for any estimator, in addition to the estimator's registered metrics."""
    from ..utils.references import load_object

    fns = []
    for ref in refs or []:
        fn = load_object(str(ref))
        if not callable(fn):
            raise TypeError(f"algorithm.metrics entry {ref!r} is not callable.")
        fns.append(fn)
    return fns
