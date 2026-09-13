"""Agent RL algorithms: advantage estimators for AgentFly's training batches.

One module per algorithm family; each registers a function per batch layout it supports
(see :mod:`.registry`). The trainer resolves ``algorithm.adv_estimator`` + the batch's
``meta_info["layout"]`` here before falling back to verl's own estimators.

Adding an algorithm: create ``agentfly/algorithms/<name>.py`` with
``@register_estimator("<name>", layout=...)`` functions and import it below.
"""

from .registry import (
    ESTIMATOR_REGISTRY,
    LAYOUTS,
    estimator_layouts,
    get_estimator,
    has_estimator,
    register_estimator,
)
from . import gigpo  # noqa: E402,F401  (registers the GiGPO estimators)

__all__ = [
    "ESTIMATOR_REGISTRY",
    "LAYOUTS",
    "estimator_layouts",
    "get_estimator",
    "has_estimator",
    "register_estimator",
]
