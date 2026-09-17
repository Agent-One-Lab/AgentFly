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
from .opd import (  # noqa: E402  (teacher step, OPD estimator + metrics)
    compute_opd_metrics,
    compute_teacher_log_probs,
)
from .teacher import (  # noqa: E402  (teacher-scoring client for on-policy distillation)
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
    "ScoreStats",
    "SequenceTooLongError",
    "TeacherClient",
    "TeacherEndpoints",
    "TeacherError",
    "TeacherIdentityError",
    "TeacherUnavailableError",
    "compute_opd_metrics",
    "compute_teacher_log_probs",
    "estimator_layouts",
    "get_estimator",
    "has_estimator",
    "register_estimator",
]
