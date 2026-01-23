import os

"""
Set the environment variables here.
"""

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Get parent folder
AGENT_HOME = os.path.dirname(__file__).split("/")[:-3]

AGENT_HOME = "/".join(AGENT_HOME)

AGENT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

USER_CACHE_DIR = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))

AGENT_CACHE_DIR = os.path.join(USER_CACHE_DIR, "AgentFly")

AGENT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

ENROOT_HOME = os.path.join(AGENT_CACHE_DIR, "enroot")

if not os.path.exists(os.path.join(ENROOT_HOME, "images")):
    os.makedirs(os.path.join(ENROOT_HOME, "images"))

AGENT_DATA_DIR = os.getenv("AGENT_DATA_DIR", AGENT_DATA_DIR)

AGENT_CONFIG_DIR = os.getenv("AGENT_CONFIG_DIR", AGENT_CONFIG_DIR)

ENROOT_HOME = os.getenv("ENROOT_HOME", ENROOT_HOME)

if os.getenv("ENROOT_DEBUG", "0") == "1":
    ENROOT_DEBUG = True
else:
    ENROOT_DEBUG = False

os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
