# Apply Enroot docker patch first so swebench/swesmith use enroot when loaded.
try:
    from .swe_rewards import swesmith_patch  # noqa: F401
except ImportError:
    pass

from .alfworld_reward import alfworld_episode_reward
from .code_reward import code_reward_test
from .gui_reward import gui_reward
from .math_reward import (
    math_equal_reward,
    math_equal_reward_think,
    math_equal_reward_tool,
    math_string_equal_reward_tool,
)
from .qa_reward import qa_em_reward, qa_f1_reward, qa_f1_reward_tool
from .reward_base import (
    BaseReward,
    get_reward_from_name,
    get_rewards_from_names,
    list_available_rewards,
    register_reward,
    reward,
)
from .types import RewardResult, RewardReturn
from .scienceworld_reward import scienceworld_reward
from .webshop_reward import webshop_reward
from .swe_rewards.swe_rewards import swe_reward, r2e_gym_reward
from .vlm_as_judge.simuscene_reward import vlm_as_judge_pass_reward, vlm_as_judge_pass_reward_multi_model

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
    "alfworld_episode_reward",
    "scienceworld_reward",
    "gui_reward",
    "code_reward_test",
    "vlm_as_judge_pass_reward",
    "vlm_as_judge_pass_reward_multi_model",
]
