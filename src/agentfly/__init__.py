import os
import shutil

__version__ = "0.2.0"

"""
Set the environment variables here.
"""

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


# Tool execution: if True (default), exceptions inside a tool's user function are turned
# into observation strings for the LLM. If False, those exceptions propagate (fail-fast).
# Env TOOL_ERROR_AS_OBSERVATION: unset defaults to True; must be "0" or "1" when set.
if os.getenv("TOOL_ERROR_AS_OBSERVATION", "1") == "1":
    TOOL_ERROR_AS_OBSERVATION = True
else:
    TOOL_ERROR_AS_OBSERVATION = False

os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
