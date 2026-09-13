import traceback

from ....core import Context
from ....envs.alfworld_env import ALFWorldSpec, _scalar
from ...decorator import tool

# verl-agent's ALFWorld reward: 10 * won (agent_system/.../alfworld/envs.py:compute_reward).
# The raw TextWorld env reward is ignored; the per-step reward is 10 on the winning
# transition, 0 otherwise. The episode reward (alfworld_episode_reward) is the same 10 * won.
ALFWORLD_WON_REWARD = 10.0


async def _get_alfworld_env(context: Context):
    """Acquire the ALFWorld resource, resetting it to the episode's task on first use."""
    need_reset = not context.is_spec_acquired(ALFWorldSpec)
    env = await context.acquire_resource(
        spec=ALFWorldSpec,
        scope="global",
        backend="local",
    )
    if need_reset:
        meta = context.metadata or {}
        env_args = {"task_id": meta["task_id"]} if "task_id" in meta else None
        await env.reset(env_args=env_args, split=meta.get("split", "train"))
    return env


def format_observation(observation: str, commands) -> str:
    """Append the admissible-action menu to the observation so the model can pick
    a valid action every turn (mirrors verl-agent's per-step prompt) without
    spending a turn on a separate lookup tool. Shared with the agent's first-node
    hook so the initial reset observation is rendered the same way as step results."""
    text = str(observation)
    if commands:
        text += "\n\nAdmissible actions: [" + ", ".join(commands) + "]"
    return text


@tool(
    name="alfworld_step",
    description="Take an action in the ALFWorld environment and return the resulting observation plus the admissible actions for the new state.",
    stateful=True,
)
async def alfworld_step(action: str, context: Context):
    """
    Take an action in the ALFWorld environment.

    Args:
        action (str): The action to take in the environment.
        context (Context): Injected rollout context; used to acquire the ALFWorld resource.

    Returns:
        dict: ``observation`` (raw text + admissible actions) shown to the model,
        the raw ``anchor`` (undecorated observation) used as the step-grouping key,
        the first-class per-step ``step_reward`` (the RAW env reward), and ``done`` /
        ``invalid_action`` metrics.

    ``step_reward`` is the **raw** ALFWorld env reward (sparse: 0 until the winning
    step). The invalid-action penalty is NOT folded in here — it is applied by the
    GiGPO estimator *after* discounting, as a local per-step deduction
    (``return[t] -= coef * invalid[t]``), matching verl-agent's
    ``apply_invalid_action_penalty`` (``invalid_action_penalty_coef=0.1``). Folding
    it into ``step_reward`` before discounting (the old behavior) let future
    penalties propagate backward and dominate the discounted return of failing
    trajectories, drowning the sparse terminal-success signal — so the step term
    credited turn-efficiency instead of success. The ``invalid_action`` flag in
    ``metrics`` is what carries the signal downstream (harvested per turn into the
    trainer batch as ``step_invalids``).
    """
    try:
        env = await _get_alfworld_env(context)
        # verl-agent's alfworld_projection lowercases the extracted action before stepping
        # (``extracted_action.strip().lower()``); the env's admissible commands are lowercase,
        # so a capitalized action would otherwise miss the admissible set and read as invalid.
        obs, reward, done, info = await env.step(action.strip().lower())
        commands = info.get("admissible_commands") if info else None
        invalid = str(obs).strip().lower().startswith("nothing happens")
        won = bool(_scalar(info.get("won", False))) if info else False
        lost = bool(_scalar(info.get("lost", False))) if info else False
        # Episode end is signalled by the task actually RESOLVING (won/lost), NOT by the env's
        # ``done``. This env build's ``done`` is a broken constant — it reads True from step 1
        # while the game keeps functioning (verified by direct probe), so honoring it would end
        # every episode after one step. ``won`` is reliable in the same info dict (it drives the
        # reward and reads correctly). Unresolved episodes run to the rollout's max_turns, which
        # equals verl-agent's env.max_steps (50). See resolved-vs-done note above.
        resolved = won or lost
        return {
            "observation": format_observation(obs, commands),
            "anchor": obs,  # raw state (no admissible-action menu) for GiGPO grouping
            # Explicit ``end`` control on task resolution → the rollout ends the episode
            # (both chain and step honor it). ``None`` while live leaves control to the policy.
            "control": "end" if resolved else None,
            # verl-agent reward = 10 * won (raw env reward ignored). Sparse: 10 on the
            # winning step, 0 otherwise. Penalty is applied post-discount by the estimator.
            "step_reward": ALFWORLD_WON_REWARD * float(won),
            "metrics": {
                # Trustworthy episode-resolution flag (won/lost), not the env's broken done.
                "done": float(resolved),
                "won": float(won),
                "lost": float(lost),
                "invalid_action": float(invalid),
                # Raw admissible-command list (not the joined menu) so the flat verl-agent
                # prompt builder can format it its own way. Non-numeric → skipped by metric
                # averaging; carried on the tool result for the rollout to read.
                "admissible_actions": list(commands) if commands else [],
            },
        }
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    print("ALFWorld Tools Schema:")
    print("======================")
    print("alfworld_step schema:")
    print(alfworld_step.schema)
