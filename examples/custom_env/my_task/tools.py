"""The guessing tool and its reward. Both use the same environment instance through the context."""

# --8<-- [start:get_env]
from agentfly.core import Context
from agentfly.rewards import reward
from agentfly.tools import tool

from .env import GuessNumberSpec


async def get_env(context: Context):
    """This rollout's game, set up on first use from the task's ``target`` field."""
    first_use = not context.is_spec_acquired(GuessNumberSpec)
    env = await context.acquire_resource(spec=GuessNumberSpec, scope="global")
    if first_use:  # a pooled instance still holds the previous rollout's game
        await env.reset(target=context.metadata["target"])
    return env


# --8<-- [end:get_env]


# --8<-- [start:tool_reward]
@tool(name="guess", description="Guess the secret number. Returns 'higher', 'lower' or 'correct'.")
async def guess(number: int, context: Context) -> str:
    """
    Args:
        number (int): Your guess.
    """
    env = await get_env(context)
    return await env.step(str(number))


@reward(name="guess_reward")
async def guess_reward(final_response: str, context: Context) -> dict:
    """1 if the agent found the number; ``guesses`` is logged as an extra metric."""
    env = await get_env(context)
    return {"reward": 1.0 if env.solved else 0.0, "guesses": env.guesses}


# --8<-- [end:tool_reward]
