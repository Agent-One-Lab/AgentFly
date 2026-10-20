"""Environments that stateful tools and rewards interact with.

Container-backed environments (Python sandbox, ALFWorld, WebShop, ScienceWorld), each with a
ready-made resource spec (``PythonSandboxSpec``, ``ALFWorldSpec``, ...) to pass to
``context.acquire_resource``; an in-process chess-puzzle environment; and the warm-pool manager
used to share environments.
"""

from .alfworld_env import ALFWorldEnv, ALFWorldSpec
from .chess_env import ChessPuzzleEnv
from .manager import EnvironmentManager, WarmPool
from .python_env import PythonSandboxEnv, PythonSandboxSpec
from .redis_env import RedisEnv
from .scienceworld_env import ScienceWorldEnv, ScienceWorldSpec
from .webshop_text_env import WebShopEnv, WebShopSpec

__all__ = [
    "PythonSandboxEnv",
    "ALFWorldEnv",
    "WebShopEnv",
    "ScienceWorldEnv",
    "PythonSandboxSpec",
    "ALFWorldSpec",
    "WebShopSpec",
    "ScienceWorldSpec",
    "RedisEnv",
    "ChessPuzzleEnv",
    "WarmPool",
    "EnvironmentManager",
]
