"""The ``agentfly`` command line: ``train``, ``deploy``, ``swebench`` and ``search``.

Each subcommand forwards its remaining arguments to the module that implements it (Hydra
overrides for ``train``). ``agentfly train`` first removes the node's stale ``agentfly-*``
enroot containers unless ``AF_ENROOT_CLEANUP_ON_START=0``.
"""

import os
import sys
from importlib import import_module

from . import configure_logging
from .templates import *  # noqa: F403


def main():
    """
    Main entry point for the AgentFly CLI.
    This is a pass-through dispatcher that respects Hydra's argument parsing.
    """

    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
        print("Usage: agentfly <command> [args...]")
        print("\nAvailable commands:")
        print("  train    - Run the PPO training script with Hydra arguments.")
        print("  deploy   - Print a vLLM serve command for local inference.")
        print("  swebench - Run an SWE-Bench-style agent over a JSON dataset.")
        print("  search   - Start the dense-retriever HTTP server.")
        sys.exit(0)

    command = sys.argv[1]
    if command == "train":
        help_requested = any(arg in ("-h", "--help") for arg in sys.argv[2:])
        if not help_requested:  # --help must not touch the node's containers
            from .resources.containers import clear_agentfly_containers

            clear_agentfly_containers()
        try:
            target_module = import_module(".verl.trainer.main_ppo", package="agentfly")
            from .verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
        except ImportError as e:
            print(
                f"agentfly train needs the training extra ({e}). Install it with "
                '`uv sync --extra verl` (source checkout) or `uv pip install "agentfly-rl[verl]"`.',
                file=sys.stderr,
            )
            sys.exit(1)

        # Training logs show AgentFly's INFO messages (per-batch summaries, prompt checks): here,
        # and in the Ray workers, which import agentfly with AF_LOG_LEVEL set.
        level = os.environ.get("AF_LOG_LEVEL", "INFO")
        configure_logging(level)
        PPO_RAY_RUNTIME_ENV["env_vars"].setdefault("AF_LOG_LEVEL", level)

        # Rewrite sys.argv for the Hydra script. Hydra expects the first
        # element to be the script path, followed by its own arguments.
        sys.argv = [target_module.__file__] + sys.argv[2:]

        target_module.main()
    elif command == "deploy":
        target_module = import_module(".utils.deploy", package="agentfly")
        sys.argv = [target_module.__file__] + sys.argv[2:]
        target_module.main()
    elif command == "search":
        target_module = import_module(".tools.src.search.retriever_server", package="agentfly")
        sys.argv = [target_module.__file__] + sys.argv[2:]
        target_module.main()
    elif command == "swebench":
        target_module = import_module(".utils.swebench", package="agentfly")
        sys.argv = [target_module.__file__] + sys.argv[2:]
        target_module.main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
