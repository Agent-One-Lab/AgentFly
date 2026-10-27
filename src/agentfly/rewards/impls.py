"""Built-in reward implementations, imported lazily.

These modules pull heavy, backend-specific dependencies (sympy/math_verify,
openai, cv2, enroot, ...) and register their rewards via the ``@reward``
decorator as a side effect of import. They are deliberately NOT imported by
``rewards/__init__`` at package-load time — a bare ``import agentfly.rewards``
(and therefore ``import agentfly.agents``) must stay cheap.

This module is imported on demand by ``rewards/__init__.__getattr__`` (attribute
access like ``from agentfly.rewards import alfworld_reward``) and by
``rewards.reward_base.ensure_builtins_loaded`` (name lookup via
``get_reward_from_name`` / ``list_available_rewards``). Importing it once
registers every built-in reward and binds every built-in name into this module's
namespace, which ``__init__`` re-exports.
"""

# Apply the Enroot docker patch first so swebench/swesmith use enroot when loaded.
try:
    from .swe_rewards import swesmith_patch  # noqa: F401
except ImportError:
    pass

from .alfworld_rewards import alfworld_reward
from .chess_rewards import chess_puzzle_reward, chess_puzzle_reward_simple
from .gui_rewards import gui_reward
from .math_rewards import (
    math_equal_reward,
    math_equal_reward_think,
    math_equal_reward_tool,
    math_reward_thought_with_tool,
    math_string_equal_reward_tool,
)
from .qa_rewards import (
    infoseek_reward,
    ok_vqa_reward,
    qa_em_format_reward,
    qa_em_reward,
    qa_em_reward_tool_call,
    qa_f1_reward,
    qa_f1_reward_tool,
)
from .scienceworld_rewards import scienceworld_reward
from .swe_rewards.swe_rewards import r2e_gym_reward, swe_reward
from .vlm_as_judge.simuscene_rewards import (
    llm_as_judge_pass_reward_multi_model,
    vlm_as_judge_pass_reward,
    vlm_as_judge_pass_reward_multi_model,
    vlm_as_judge_ratio_reward_multi_model,
)
from .webshop_rewards import webshop_reward, webshop_score_reward

# Every built-in, re-exported by the package ``__init__``. Listing them here also keeps
# linters from deleting these imports as unused, which would unregister the built-ins.
__all__ = [
    "alfworld_reward",
    "chess_puzzle_reward",
    "chess_puzzle_reward_simple",
    "gui_reward",
    "infoseek_reward",
    "llm_as_judge_pass_reward_multi_model",
    "math_equal_reward",
    "math_equal_reward_think",
    "math_equal_reward_tool",
    "math_reward_thought_with_tool",
    "math_string_equal_reward_tool",
    "ok_vqa_reward",
    "qa_em_format_reward",
    "qa_em_reward",
    "qa_em_reward_tool_call",
    "qa_f1_reward",
    "qa_f1_reward_tool",
    "r2e_gym_reward",
    "scienceworld_reward",
    "swe_reward",
    "vlm_as_judge_pass_reward",
    "vlm_as_judge_pass_reward_multi_model",
    "vlm_as_judge_ratio_reward_multi_model",
    "webshop_reward",
    "webshop_score_reward",
]
