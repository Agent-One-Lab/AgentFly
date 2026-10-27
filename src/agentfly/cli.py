"""The ``agentfly`` command line: ``train``, ``deploy``, ``swebench``, ``search`` and ``pull-images``.

Each subcommand forwards its remaining arguments to the module that implements it (Hydra
overrides for ``train``). ``agentfly train`` first removes this machine's AgentFly containers whose owner process has
ended (see :func:`~agentfly.resources.containers.sweep_orphaned_containers`).
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
        print("  deploy   - Start a vLLM OpenAI-compatible server for a model.")
        print("  swebench - Run an SWE-Bench-style agent over a JSON dataset.")
        print("  search   - Start the dense-retriever HTTP server.")
        print("  pull-images - Pre-pull the enroot images a dataset needs (SWE tasks).")
        sys.exit(0)

    command = sys.argv[1]
    if command == "train":
        help_requested = any(arg in ("-h", "--help") for arg in sys.argv[2:])
        if not help_requested:  # --help must not touch the node's containers
            from .resources.containers import sweep_orphaned_containers

            removed = sweep_orphaned_containers()
            print(f"[agentfly] removed {len(removed)} orphaned container(s)")
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
        target_module.main(prog_name="agentfly deploy")
    elif command == "search":
        target_module = import_module(".tools.src.search.retriever_server", package="agentfly")
        sys.argv = [target_module.__file__] + sys.argv[2:]
        target_module.main()
    elif command == "pull-images":
        from .utils.pull_images import main as pull_images

        sys.exit(pull_images(sys.argv[2:]))
    elif command == "swebench":
        target_module = import_module(".utils.swebench", package="agentfly")
        sys.argv = [target_module.__file__] + sys.argv[2:]
        target_module.main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
