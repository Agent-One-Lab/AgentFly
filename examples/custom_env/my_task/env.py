"""A number-guessing environment: the agent guesses a secret number and is told higher or lower."""

# --8<-- [start:env]
from agentfly.envs.env_base import BaseEnv
from agentfly.resources import LocalEnvResourceSpec


class GuessNumberEnv(BaseEnv):
    """State for one game. AgentFly creates, pools and reuses instances; tools call ``step``."""

    def __init__(self, low: int = 1, high: int = 100):
        self.low, self.high = low, high
        self.target = None
        self.guesses = 0
        self.solved = False

    async def start(self) -> None:
        """Called once, when the engine creates the instance (start processes, open clients...)."""

    async def reset(self, target: int) -> str:
        """Start a new game. Pooled instances are reused, so this must clear all old state."""
        self.target, self.guesses, self.solved = target, 0, False
        return f"New game: the number is between {self.low} and {self.high}."

    async def step(self, action: str) -> str:
        guess = int(action)
        self.guesses += 1
        if guess == self.target:
            self.solved = True
            return "correct"
        return "higher" if guess < self.target else "lower"

    async def aclose(self) -> None:
        """Called when the instance is destroyed (stop processes, close clients...)."""


# How the engine creates the env: an importable dotted path, its constructor arguments,
# and the most instances alive at once (concurrent rollouts wait for a free one).
GuessNumberSpec = LocalEnvResourceSpec(
    env_cls_path="my_task.env.GuessNumberEnv",
    init_kwargs={"low": 1, "high": 100},
    max_global_num=256,  # train.sh runs 32 tasks x 8 chains at once
)
# --8<-- [end:env]
