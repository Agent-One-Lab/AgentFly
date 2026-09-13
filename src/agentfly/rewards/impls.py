"""Built-in reward implementations, imported lazily.

These modules pull heavy, backend-specific dependencies (sympy/math_verify,
openai, cv2, enroot, ...) and register their rewards via the ``@reward``
decorator as a side effect of import. They are deliberately NOT imported by
``rewards/__init__`` at package-load time — a bare ``import agentfly.rewards``
(and therefore ``import agentfly.agents``) must stay cheap.

This module is imported on demand by ``rewards/__init__.__getattr__`` (attribute
access like ``from agentfly.rewards import alfworld_episode_reward``) and by
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

from .alfworld_reward import alfworld_episode_reward
from .chess_reward import chess_puzzle_reward, chess_puzzle_reward_simple
from .code_reward import code_reward_test
from .gui_reward import gui_reward
from .math_reward import (
    math_equal_reward,
    math_equal_reward_think,
    math_equal_reward_tool,
    math_string_equal_reward_tool,
)
from .qa_reward import qa_em_reward, qa_f1_reward, qa_f1_reward_tool
from .scienceworld_reward import scienceworld_reward
from .webshop_reward import webshop_episode_reward, webshop_reward
from .swe_rewards.swe_rewards import swe_reward, r2e_gym_reward
from .vlm_as_judge.simuscene_reward import (
    vlm_as_judge_pass_reward,
    vlm_as_judge_pass_reward_multi_model,
)
