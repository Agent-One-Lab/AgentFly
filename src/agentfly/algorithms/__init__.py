"""Agent RL algorithms: advantage estimators for AgentFly's training batches.

One module per algorithm family; each registers a function per batch layout it supports
(see :mod:`.registry`). The trainer resolves ``algorithm.adv_estimator`` + the batch's
``meta_info["layout"]`` here before falling back to verl's own estimators.

Adding an algorithm: create ``agentfly/algorithms/<name>.py`` with
``@register_estimator("<name>", layout=...)`` functions and import it below. An algorithm
defined outside AgentFly needs no edit here: set ``algorithm.adv_estimator`` to its import
reference (``"pkg.mod:fn"`` or ``"/path/file.py:fn"``; see :func:`.registry.resolve_estimator`).
"""

from . import gigpo  # noqa: F401  (registers the GiGPO estimators)
from .opd import (  # teacher step, OPD estimator + metrics
    compute_opd_metrics,
    compute_teacher_log_probs,
)
from .registry import (
    ESTIMATOR_REGISTRY,
    LAYOUTS,
    METRICS_REGISTRY,
    compute_estimator_metrics,
    estimator_layouts,
    get_estimator,
    has_estimator,
    load_metrics,
    register_estimator,
    register_metrics,
    resolve_estimator,
)
from .teacher import (  # teacher-scoring client for on-policy distillation
    ScoreStats,
    SequenceTooLongError,
    TeacherClient,
    TeacherEndpoints,
    TeacherError,
    TeacherIdentityError,
    TeacherUnavailableError,
)

__all__ = [
    "ESTIMATOR_REGISTRY",
    "LAYOUTS",
    "METRICS_REGISTRY",
    "ScoreStats",
    "SequenceTooLongError",
    "TeacherClient",
    "TeacherEndpoints",
    "TeacherError",
    "TeacherIdentityError",
    "TeacherUnavailableError",
    "compute_estimator_metrics",
    "compute_opd_metrics",
    "compute_teacher_log_probs",
    "estimator_layouts",
    "get_estimator",
    "has_estimator",
    "load_metrics",
    "register_estimator",
    "register_metrics",
    "resolve_estimator",
]
