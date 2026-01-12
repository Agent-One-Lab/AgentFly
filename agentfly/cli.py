import sys
import os
from importlib import import_module


def main():
    """
    Main entry point for the AgentFly CLI.
    This is a pass-through dispatcher that respects Hydra's argument parsing.
    """

    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
        print("Usage: agentfly <command> [args...]")
        print("\nAvailable commands:")
        print("  train    - Run the PPO training script with Hydra arguments.")
        print("  deploy   - Deploy a service (placeholder).")
        sys.exit(0)

    command = sys.argv[1]
    if command == 'train':
        
        target_module = import_module(".verl.trainer.main_ppo", package="agentfly")

        # Rewrite sys.argv for the Hydra script. Hydra expects the first
        # element to be the script path, followed by its own arguments.
        sys.argv = [target_module.__file__] + sys.argv[2:]
        
        target_module.main()
    elif command == "deploy":
        target_module = import_module(".utils.deploy", package="agentfly")
        sys.argv = [target_module.__file__] + sys.argv[2:]
        target_module.main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()