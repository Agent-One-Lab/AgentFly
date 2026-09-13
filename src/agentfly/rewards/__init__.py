"""AgentFly rewards package.

The light core (the ``@reward`` decorator, the registry accessors, ``BaseReward``
and the result types) is imported eagerly. The built-in reward *implementations*
live in :mod:`.impls` and are imported lazily — they pull heavy deps
(sympy/math_verify, openai, cv2, enroot, ...), so a bare ``import
agentfly.rewards`` (and hence ``import agentfly.agents``) stays cheap. The impls
load on first access, whether by attribute (``from agentfly.rewards import
alfworld_episode_reward``) or by name (``get_reward_from_name`` /
``list_available_rewards``).
"""

from typing import TYPE_CHECKING

from .reward_base import (
    BaseReward,
    get_reward_from_name,
    get_rewards_from_names,
    list_available_rewards,
    register_reward,
    reward,
)
from .types import RewardResult, RewardReturn

if TYPE_CHECKING:
    # Static tooling cannot discover exports through __getattr__. Keep these
    # declarations in sync with the lazy public exports in __all__; importing
    # them only for type checking preserves lazy loading at runtime.
    from .impls import (
        alfworld_episode_reward,
        code_reward_test,
        gui_reward,
        math_equal_reward,
        math_equal_reward_think,
        math_equal_reward_tool,
        math_string_equal_reward_tool,
        qa_em_reward,
        qa_f1_reward,
        qa_f1_reward_tool,
        r2e_gym_reward,
        scienceworld_reward,
        swe_reward,
        vlm_as_judge_pass_reward,
        vlm_as_judge_pass_reward_multi_model,
        webshop_episode_reward,
        webshop_reward,
    )


def __getattr__(name):
    # Lazy attribute access for the built-in reward impls (everything in the impls
    # module). Route through the registry loader first so user-vs-builtin name
    # collisions resolve consistently (user wins), then read the name off it. Guard
    # dunder names and the loader module name so importing ``impls`` — which itself
    # resolves ``rewards.impls`` — can't recurse back into here.
    if name == "impls" or name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .reward_base import ensure_builtins_loaded

    ensure_builtins_loaded()
    from . import impls

    try:
        value = getattr(impls, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    globals()[name] = value  # cache so __getattr__ isn't hit again for this name
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "BaseReward",
    "RewardResult",
    "RewardReturn",
    "get_reward_from_name",
    "get_rewards_from_names",
    "list_available_rewards",
    "register_reward",
    "reward",
    "swe_reward",
    "r2e_gym_reward",
    "qa_em_reward",
    "qa_f1_reward",
    "qa_f1_reward_tool",
    "math_equal_reward",
    "math_equal_reward_tool",
    "math_equal_reward_think",
    "math_string_equal_reward_tool",
    "webshop_reward",
    "webshop_episode_reward",
    "alfworld_episode_reward",
    "scienceworld_reward",
    "gui_reward",
    "code_reward_test",
    "vlm_as_judge_pass_reward",
    "vlm_as_judge_pass_reward_multi_model",
]
