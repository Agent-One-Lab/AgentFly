from .swesmith_patch import evaluate_swesmith
from ...core import Context
from ...resources import ResourceSpec
from .utils import get_patch_from_runtime
from ...rewards.reward_base import reward
from .r2e_gym.eval import reward_from_container
from .r2e_gym.eval import setup_container_for_reward
import asyncio

# asyncio.to_thread only prevents blocking the event loop; it does not shorten wall-clock
# time. Within one reward the steps are inherently sequential (need patch before evaluate).
# For parallelism: use num_chains > 1 so multiple rewards run concurrently across rollouts.

# Default timeout (seconds) for evaluate_swesmith; override via context.metadata["eval_timeout"].
DEFAULT_EVAL_TIMEOUT = 300

@reward(name="r2e_gym_reward")
async def r2e_gym_reward(context: Context) -> dict:
    image_id = context.metadata.get("image_id")
    if not image_id:
        return "Error: context.metadata['image_id'] is required for shell tool."
    rollout_id = context.rollout_id
    spec = ResourceSpec(
        category="container",
        image=image_id,
    )

    container = await context.acquire_resource(
        spec=spec, id=rollout_id, scope="rollout", backend="local"
    )

    # Allow configuring evaluation timeout via context.metadata["eval_timeout"].
    eval_timeout = context.metadata.get("eval_timeout", DEFAULT_EVAL_TIMEOUT)

    try:
        # Setup phase: create /run_tests.sh and related symlinks inside the container.
        # Use a bounded timeout so container setup cannot hang indefinitely.
        await asyncio.wait_for(
            asyncio.to_thread(
                setup_container_for_reward,
                container._container,
                dataset="r2e",
                setup_timeout=eval_timeout,
            ),
            timeout=eval_timeout + 10,
        )
    except asyncio.TimeoutError:
        return {"reward": 0.0, "eval_timeout": 1.0, "out": ""}

    try:
        # Test run + grading. reward_from_container applies timeout to the inner
        # test command; wait_for bounds the overall wall-clock time from the
        # trainer's point of view.
        reward, out = await asyncio.wait_for(
            asyncio.to_thread(
                reward_from_container,
                container._container,
                context.metadata,
                dataset="r2e",
                timeout=eval_timeout,
                get_test_output=True,
            ),
            timeout=eval_timeout + 10,
        )
    except asyncio.TimeoutError:
        return {"reward": 0.0, "eval_timeout": 1.0, "out": ""}

    return {"reward": reward, "eval_timeout": 0.0, "out": out}



@reward(name="swe_reward")
async def swe_reward(context: Context) -> dict:
    image_id = context.metadata.get("image_id")
    if not image_id:
        return "Error: context.metadata['image_id'] is required for shell tool."
    rollout_id = context.rollout_id
    spec = ResourceSpec(
        category="container",
        image=image_id,
    )
    container = await context.acquire_resource(
        spec=spec, id=rollout_id, scope="rollout", backend="local"
    )
    eval_timeout = context.metadata.get("eval_timeout", DEFAULT_EVAL_TIMEOUT)
    git_patch = await get_patch_from_runtime(
        container,
        dataset="swe-smith",
        workspace_dir_name="/testbed",
        timeout=eval_timeout,
    )
    if not (git_patch and git_patch.strip()):
        # No edits or extraction failed: no patch to evaluate.
        return {"reward": 0.0, "eval_timeout": 0.0}
    try:
        # Run in thread so event loop is not blocked; evaluation is still slow (test run).
        result = await asyncio.wait_for(
            asyncio.to_thread(
                evaluate_swesmith,
                sample=context.metadata,
                patch=git_patch,
                container=container._container,
            ),
            timeout=eval_timeout,
        )
    except asyncio.TimeoutError:
        return {"reward": 0.0, "eval_timeout": 1.0}

    if result["resolved"]:
        return {"reward": 1.0, "eval_timeout": 0.0}
    else:
        return {"reward": 0.0, "eval_timeout": 0.0}


