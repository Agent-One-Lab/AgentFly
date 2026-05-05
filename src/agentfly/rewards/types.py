"""Typed result and return-shape aliases for reward functions.

User-authored ``@reward`` functions return ``float`` or ``dict``; the
framework normalizes those into a :class:`RewardResult` at the call site
(see ``reward_base.calculate_reward``) so downstream code reads
``result.reward`` (always ``float``) and ``result.extras`` (always
``Dict[str, Any]``). The :class:`RewardReturn` ``TypedDict`` is an
annotation alias only — no runtime effect.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, TypedDict, Union


@dataclass
class RewardResult:
    """Normalized output of a reward function call.

    The framework constructs these via :meth:`from_raw` from whatever the
    user's ``@reward`` function returned. ``reward`` is the scalar used by
    the trainer; ``extras`` carries any additional logged metrics the
    function returned (e.g. ``f1``, ``em``, ``precision``).
    """

    reward: float
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        raw: Union[float, int, dict, "RewardResult", None],
    ) -> "RewardResult":
        """Normalize a reward function's raw return into a ``RewardResult``.

        Accepts:
        - ``float`` / ``int``: ``reward`` set; ``extras`` empty.
        - ``dict``: must have a ``reward`` key. All other keys go into ``extras``.
        - ``RewardResult``: returned as-is.
        - ``None``: returns a zero-reward result with empty extras (matches
          the historical behavior when no reward function was set).
        """
        if raw is None:
            return cls(reward=0.0, extras={})
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, bool):
            # Avoid bool->float coercion silently swallowing typing mistakes.
            return cls(reward=float(raw), extras={})
        if isinstance(raw, (float, int)):
            return cls(reward=float(raw), extras={})
        if isinstance(raw, dict):
            if "reward" not in raw:
                raise ValueError(
                    "reward key required when reward fn returns a dict; "
                    f"got keys: {list(raw.keys())}"
                )
            reward = float(raw["reward"])
            extras = {k: v for k, v in raw.items() if k != "reward"}
            return cls(reward=reward, extras=extras)
        raise ValueError(
            f"reward fn returned {type(raw).__name__}; "
            "expected float, int, dict, RewardResult, or None."
        )


class RewardReturn(TypedDict, total=False):
    """Annotation alias for what a user ``@reward`` function returns.

    Rewards may return a bare ``float`` **or** a dict with ``reward`` plus
    arbitrary extra scalar metric keys. The ``reward`` key is required if a
    dict is returned; any other keys flow through to :attr:`RewardResult.extras`.
    """

    reward: float
