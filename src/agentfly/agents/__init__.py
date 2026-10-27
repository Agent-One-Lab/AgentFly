"""Agents: the agent classes, :class:`AutoAgent`, and the result types of a run.

:class:`BaseAgent` runs multi-turn, multi-chain rollouts with tools and a reward; the
specialized agents differ in how they prompt the model and parse its tool calls (native tool
calling, ReAct, the ``<action>`` format, code, GUI actions, SWE tasks, image editing).
:class:`AutoAgent` builds an agent from a config.
"""

from ..utils.llm_backends import (
    AsyncVerlBackend,
    AsyncVLLMBackend,
    ClientBackend,
    ClientConfig,
)
from .agent_base import BaseAgent
from .auto import AutoAgent
from .specialized.action_agent import ActionAgent
from .specialized.code_agent import CodeAgent
from .specialized.gui_agent import GUIAgent
from .specialized.hf_agent import HFAgent, SearchR1Agent
from .specialized.image_agent import ImageEditingAgent
from .specialized.miniswe import MinisweAgent
from .specialized.react_agent import ReactAgent
from .specialized.swe_agents import BashSWEAgent, FunctionCallSWEAgent, Qwen3CoderSWEAgent
from .types import RunResult, Segment, Trajectory

__all__ = [
    "BaseAgent",
    "AutoAgent",
    "ReactAgent",
    "CodeAgent",
    "BashSWEAgent",
    "FunctionCallSWEAgent",
    "Qwen3CoderSWEAgent",
    "SearchR1Agent",
    "GUIAgent",
    "HFAgent",
    "ImageEditingAgent",
    "ActionAgent",
    "MinisweAgent",
    "Segment",
    "Trajectory",
    "RunResult",
    "ClientBackend",
    "ClientConfig",
    "AsyncVLLMBackend",
    "AsyncVerlBackend",
]
