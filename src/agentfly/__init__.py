"""AgentFly: agent rollouts and reinforcement learning for LLM agents with tools and resources.

Importing the package sets up its per-user directories and container settings, each
overridable with an environment variable of the same name:

- ``AF_HOME`` (``~/.agentfly``): the root of AgentFly's state.
- ``AF_CACHE_DIR`` / ``AF_DATA_DIR``: downloaded tool data, enroot images and datasets.
- ``AF_CONFIG_DIR``: user-editable config, seeded from the packaged defaults on first run.
- ``ENROOT_HOME``: enroot's data directory (images under ``images/``).
- ``AF_CONTAINER_HOME``: AgentFly's directory inside task containers (skills and tool helpers).

The public API lives in the subpackages: :mod:`agentfly.agents`, :mod:`agentfly.tools`,
:mod:`agentfly.rewards`, :mod:`agentfly.resources`, :mod:`agentfly.envs` and
:mod:`agentfly.algorithms`.
"""

import logging
import os
import shutil
import sys

__version__ = "0.2.0"


def configure_logging(level: str = "INFO") -> None:
    """Print AgentFly's log messages at ``level`` and above to stdout, as plain lines.

    AgentFly logs through the standard ``logging`` module (logger ``"agentfly"``) and shows only
    warnings by default. ``agentfly train`` calls this with ``"INFO"`` so training logs include the
    per-batch summaries; setting the ``AF_LOG_LEVEL`` environment variable does the same in every
    process that imports AgentFly (including Ray workers).
    """
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

# User-editable config (search.yaml API keys, redis.conf, ...). Lives under
# AF_HOME so users edit ~/.agentfly/config/* instead of files inside the
# installed package. Packaged defaults are seeded here on first run (below).
AF_CONFIG_DIR = os.getenv("AF_CONFIG_DIR", os.path.join(AF_HOME, "config"))

# Default config files shipped in the wheel; seeded into AF_CONFIG_DIR when a
# file is missing (never overwriting a user's edits).
_PACKAGED_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _seed_config_dir() -> None:
    if not os.path.isdir(_PACKAGED_CONFIG_DIR):
        return
    for root, _dirs, files in os.walk(_PACKAGED_CONFIG_DIR):
        rel = os.path.relpath(root, _PACKAGED_CONFIG_DIR)
        dest_dir = AF_CONFIG_DIR if rel == "." else os.path.join(AF_CONFIG_DIR, rel)
        for name in files:
            dest = os.path.join(dest_dir, name)
            if not os.path.exists(dest):
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(os.path.join(root, name), dest)


_seed_config_dir()

ENROOT_HOME = os.getenv("ENROOT_HOME", os.path.join(AF_CACHE_DIR, "enroot"))

if not os.path.exists(os.path.join(ENROOT_HOME, "images")):
    os.makedirs(os.path.join(ENROOT_HOME, "images"))

if os.getenv("ENROOT_DEBUG", "0") == "1":
    ENROOT_DEBUG = True
else:
    ENROOT_DEBUG = False


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
