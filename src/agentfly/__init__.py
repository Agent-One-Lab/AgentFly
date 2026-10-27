"""AgentFly: agent rollouts and reinforcement learning for LLM agents with tools and resources.

Importing the package sets up its per-user directories and container settings, each
overridable with an environment variable of the same name:

- ``AF_HOME`` (``~/.agentfly``): the root of AgentFly's state.
- ``AF_CACHE_DIR`` / ``AF_DATA_DIR``: downloaded tool data and datasets.
- ``AF_CONTAINER_HOME``: AgentFly's directory inside task containers (skills and tool helpers).

The public API lives in the subpackages: :mod:`agentfly.agents`, :mod:`agentfly.tools`,
:mod:`agentfly.rewards`, :mod:`agentfly.resources`, :mod:`agentfly.envs` and
:mod:`agentfly.algorithms`.
"""

import logging
import os
import sys

__version__ = "0.2.0"


def configure_logging(level: str = "INFO") -> None:
    """Print AgentFly's log messages at ``level`` and above to stdout, as plain lines.

    AgentFly logs through the standard ``logging`` module (logger ``"agentfly"``) and shows only
    warnings by default. ``agentfly train`` calls this with ``"INFO"`` so training logs include the
    per-batch summaries; setting the ``AF_LOG_LEVEL`` environment variable does the same in every
    process that imports AgentFly (including Ray workers).

    It also limits the ``httpx`` and ``huggingface_hub`` loggers to warnings, so a process whose
    root logger is at INFO (e.g. with ``VERL_LOGGING_LEVEL=INFO``) does not print a line for every
    Hugging Face file check.
    """
    for name in ("httpx", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logger = logging.getLogger("agentfly")
    logger.setLevel(level.upper())
    if not any(getattr(h, "_agentfly", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._agentfly = True
        logger.addHandler(handler)
    logger.propagate = False  # avoid printing twice when the application configures the root logger


if os.getenv("AF_LOG_LEVEL"):
    configure_logging(os.environ["AF_LOG_LEVEL"])

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# AgentFly home: per-user state directory, following the ~/.<tool> convention
# used by most frameworks. Cache and data live under it. Override with AF_HOME
# (or the individual dirs via AF_CACHE_DIR / AF_DATA_DIR).
AF_HOME = os.getenv("AF_HOME", os.path.expanduser("~/.agentfly"))

AF_CACHE_DIR = os.getenv("AF_CACHE_DIR", os.path.join(AF_HOME, "cache"))

AF_DATA_DIR = os.getenv("AF_DATA_DIR", os.path.join(AF_HOME, "data"))

# Container-side agentfly home: where skills + tool helpers are placed INSIDE the
# task container (the in-container analogue of ~/.agentfly). Defaults to the
# (root) container user's home — agentfly's env images run as root — and is
# overridable via AF_CONTAINER_HOME for non-root images. Derived paths:
#   - skills root  -> $AF_CONTAINER_HOME/skills   (this is SKILL_DIR)
#   - tool helpers -> $AF_CONTAINER_HOME/tools/<name>
AF_CONTAINER_HOME = os.getenv("AF_CONTAINER_HOME", "/root/.agentfly")
AF_CONTAINER_SKILLS_DIR = os.path.join(AF_CONTAINER_HOME, "skills")
AF_CONTAINER_TOOLS_DIR = os.path.join(AF_CONTAINER_HOME, "tools")

# Tool-body exceptions are surfaced as a ``status="error"`` diagnosis on the ToolResult;
# whether that continues / ends / raises is the rollout's ``on_tool_error`` run policy
# (see agentfly.agents.rollout.base.Rollout), not a global env flag.

os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
