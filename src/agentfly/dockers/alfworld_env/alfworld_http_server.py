"""
ALFWorld HTTP Server

Changes vs. the original:
  * Add a global asyncio.Lock around /reset, /step, and /close so concurrent
    requests can never race on the shared `current_env` global.
  * Close the previous `current_env` before creating a new one (this shuts
    down the TextWorld AsyncBatchEnv subprocess that init_env spawns). This
    is the dominant leak in the original code: every /reset used to leak one
    Python subprocess plus its loaded game state.
  * Cache the random-mode env per split in `split_envs` (previously declared
    but unused) so we do not rescan the dataset and rebuild the env on every
    /reset when no task_id is supplied.
  * Cache configs built for specific tasks, and deepcopy only the subtrees
    that get mutated.
  * Clear `current_obs` / `current_info` on reset entry so stale data from
    the previous episode is not retained.
  * Add a /close endpoint so the client can tell the server to tear down the
    active env when it is finished with the container.
  * Force a gc.collect() after disposing of an env to hurry along the
    subprocess termination and free the game memory promptly.
"""

import asyncio
import gc
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Import ALFWorld
try:
    import alfworld.agents.modules.generic as generic
    from alfworld.agents.environment import get_environment
except ImportError as e:
    logger.error(f"Failed to import ALFWorld: {e}")
    sys.exit(1)

app = FastAPI()

# --- Global variables for environment management ---
config: Optional[Dict] = None
current_env: Optional[Any] = None
current_obs: Optional[str] = None
current_info: Optional[Dict] = None

# Identifies which env is currently bound to `current_env`:
#   - ("random", split)           -> env cached in split_envs[split]
#   - ("task",   split, task_id)  -> per-task env (not cached, must be closed)
#   - None                        -> no env bound
current_env_key: Optional[tuple] = None

# Caches for task management
task_cache: Dict[str, Dict[str, Dict]] = {}
split_envs: Dict[str, Any] = {}   # Random-mode envs, keyed by split
task_cfg_cache: Dict[str, Dict] = {}  # Per-task config deepcopies, keyed by task_id

# Serialize all env-mutating endpoints; FastAPI handles requests concurrently,
# and our env state is a single shared global.
env_lock = asyncio.Lock()


# --- Pydantic Models ---


class ActionRequest(BaseModel):
    action: str


class ResetRequest(BaseModel):
    split: str = Field(
        "train",
        description="The data split to use ('train', 'valid_seen', 'valid_unseen').",
    )
    task_id: Optional[str] = Field(
        None, description="Optional: A specific task_id to load."
    )


# --- Helpers ---


def _safe_close(env: Any) -> None:
    """Best-effort close of a TextWorld batch env, swallowing all errors.

    AlfredTWEnv.init_env() returns a TextworldBatchGymEnv wrapping an
    AsyncBatchEnv; the async variant spawns one Python subprocess per batch
    slot. Without .close() those subprocesses linger until GC (and sometimes
    beyond), which is the main memory-leak vector.
    """
    if env is None:
        return
    for meth in ("close", "shutdown"):
        fn = getattr(env, meth, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception as e:
                logger.warning(f"env.{meth}() raised: {e}")


def _build_task_cfg(split: str, task_info: Dict) -> Dict:
    """Build (and cache) the config variant needed to load a single task."""
    task_id = task_info["task_id"]
    cached = task_cfg_cache.get(task_id)
    if cached is not None:
        return cached

    trial_dir = os.path.dirname(task_info["game_file"])
    task_type_dir = os.path.basename(os.path.dirname(trial_dir))
    task_type_slug = task_type_dir.split("-")[0]

    TASK_SLUG2ID = {
        "pick_and_place_simple": 1,
        "look_at_obj_in_light": 2,
        "pick_clean_then_place_in_recep": 3,
        "pick_heat_then_place_in_recep": 4,
        "pick_cool_then_place_in_recep": 5,
        "pick_two_obj_and_place": 6,
    }
    task_type_id = TASK_SLUG2ID[task_type_slug]

    split_to_path_key = {
        "train": "train_data_path",
        "valid_seen": "valid_seen_data_path",
        "valid_unseen": "valid_unseen_data_path",
    }

    # Only deepcopy the subtrees we mutate — the rest we alias. The env
    # keeps a reference to the dict so mutating the global would be unsafe,
    # but we do not need to deepcopy everything.
    cfg = dict(config)
    cfg["dataset"] = dict(cfg.get("dataset", {}))
    cfg["env"] = dict(cfg.get("env", {}))

    cfg["dataset"]["data_path"] = trial_dir
    cfg["dataset"]["num_train_games"] = 1
    cfg["dataset"][split_to_path_key[split]] = trial_dir
    cfg["env"]["task_types"] = [task_type_id]

    task_cfg_cache[task_id] = cfg
    return cfg


def _create_env(split: str, task_info: Optional[Dict] = None):
    """Create and init an AlfredTWEnv instance. Caller owns closing it."""
    global config

    env_type = config.get("env", {}).get("type", "AlfredTWEnv")
    EnvClass = get_environment(env_type)

    if task_info:
        cfg = _build_task_cfg(split, task_info)
        logger.info(f"Loading single task {task_info['task_id']}")
        # AlfredTWEnv expects 'train' / 'eval_in_distribution' /
        # 'eval_out_of_distribution'. 'train' is safest for single-task.
        env = EnvClass(cfg, train_eval="train")
    else:
        env = EnvClass(config, train_eval=split)

    env = env.init_env(batch_size=1)
    return env


def _dispose_current_env() -> None:
    """Close whatever env is currently bound, unless it is a cached split env."""
    global current_env, current_env_key

    if current_env is None:
        current_env_key = None
        return

    # If the active env is a cached split env, leave it open for reuse.
    if current_env_key is not None and current_env_key[0] == "random":
        current_env = None
        current_env_key = None
        return

    # Task-specific envs are never cached — close and drop.
    _safe_close(current_env)
    current_env = None
    current_env_key = None
    gc.collect()


def _acquire_env(split: str, task_info: Optional[Dict]) -> Any:
    """Return an env for (split, task_info), reusing the split cache when possible."""
    global current_env, current_env_key

    if task_info is None:
        key = ("random", split)
        env = split_envs.get(split)
        if env is None:
            env = _create_env(split, None)
            split_envs[split] = env
        current_env = env
        current_env_key = key
        return env

    key = ("task", split, task_info["task_id"])
    env = _create_env(split, task_info)
    current_env = env
    current_env_key = key
    return env


def _extract_admissible_commands(info: Dict) -> list:
    if not info or "admissible_commands" not in info:
        return []
    cmds = info["admissible_commands"]
    if isinstance(cmds, list) and len(cmds) > 0 and isinstance(cmds[0], list):
        return cmds[0]
    return cmds if isinstance(cmds, list) else []


def _scan_and_cache_tasks():
    """Scan the ALFWorld data directory and cache all task metadata by split."""
    global task_cache

    data_path = os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")
    data_path = os.path.expanduser(data_path)

    task_cache = {"train": {}, "valid_seen": {}, "valid_unseen": {}, "test_unseen": {}}

    logger.info(f"Scanning for tasks in base path: {data_path}/json_2.1.1/train")

    for split in task_cache.keys():
        split_path = os.path.join(data_path, "json_2.1.1", split)
        if not os.path.exists(split_path):
            logger.warning(f"Split directory not found: {split_path}")
            continue

        for task_type_dir in os.listdir(split_path):
            task_type_path = os.path.join(split_path, task_type_dir)
            if not os.path.isdir(task_type_path):
                continue

            task_type = task_type_dir.split("-")[0]

            for trial_dir in os.listdir(task_type_path):
                if not trial_dir.startswith("trial_"):
                    continue

                trial_path = os.path.join(task_type_path, trial_dir)
                if not os.path.isdir(trial_path):
                    continue

                task_id = trial_dir
                traj_data_path = os.path.join(trial_path, "traj_data.json")
                if os.path.exists(traj_data_path):
                    task_cache[split][task_id] = {
                        "task_id": task_id,
                        "task_type": task_type,
                        "game_file": os.path.join(trial_path, "game.tw-pddl"),
                    }

    counts = {k: len(v) for k, v in task_cache.items()}
    logger.info(f"Task cache created. Found tasks: {counts}")


# --- FastAPI Event Handlers ---


@app.on_event("startup")
async def startup_event():
    global config

    config_path = os.environ.get("ALFWORLD_CONFIG", "/srv/base_config.yaml")

    try:
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], config_path]
        try:
            config = generic.load_config()
            logger.info("ALFWorld configuration loaded successfully")
            _scan_and_cache_tasks()
        finally:
            sys.argv = original_argv
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close every env we are still holding on process exit."""
    global current_env, current_env_key, split_envs
    async with env_lock:
        _safe_close(current_env)
        current_env = None
        current_env_key = None
        for split, env in list(split_envs.items()):
            _safe_close(env)
        split_envs.clear()
        gc.collect()


# --- Endpoints ---


@app.get("/health")
async def health():
    return {"status": "ok", "service": "alfworld"}


@app.get("/available_tasks")
async def get_available_tasks(split: str = "train"):
    if split not in task_cache:
        raise HTTPException(status_code=404, detail=f"Invalid split: {split}")
    return {"tasks": list(task_cache[split].values())}


@app.post("/reset")
async def reset(request: ResetRequest):
    global current_obs, current_info

    async with env_lock:
        try:
            split = request.split
            task_id = request.task_id

            if split not in task_cache:
                raise HTTPException(status_code=400, detail=f"Invalid split: {split}")

            task_info = None
            if task_id:
                if task_id not in task_cache[split]:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Task ID '{task_id}' not found in split '{split}'.",
                    )
                task_info = task_cache[split][task_id]

            # Clear stale episode state immediately so we don't hold it through
            # the (potentially slow) env construction below.
            current_obs = None
            current_info = None

            # Dispose of the previously-bound env before creating a new one.
            # For the cached random-mode env this is a no-op; for task-specific
            # envs this closes the TextWorld AsyncBatchEnv subprocess.
            _dispose_current_env()

            env = _acquire_env(split, task_info)
            obs, info = env.reset()
            logger.info("Environment reset completed")

            current_obs = obs[0] if isinstance(obs, list) else obs
            current_info = info[0] if isinstance(info, list) else info or {}

            admissible_commands = _extract_admissible_commands(current_info)

            goal = ""
            if current_info:
                goal = (
                    current_info.get("goal")
                    or current_info.get("task_description")
                    or current_info.get("task_desc")
                    or current_info.get("description")
                    or ""
                )

            task = (
                current_info.get("extra.gamefile", current_info.get("task", "unknown"))
                if current_info
                else "unknown"
            )

            logger.info(f"Reset successful - Task: {task}, Goal: {goal}")

            return {
                "observation": str(current_obs),
                "info": {
                    "admissible_commands": admissible_commands,
                    "task": task,
                    "goal": goal,
                    "steps_taken": 0,
                },
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Reset failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: ActionRequest):
    global current_env, current_obs, current_info

    async with env_lock:
        if current_env is None:
            raise HTTPException(
                status_code=400,
                detail="Environment not initialized. Call /reset first.",
            )

        try:
            actions = [request.action]
            obs, scores, dones, infos = current_env.step(actions)

            current_obs = obs[0] if isinstance(obs, list) else obs

            raw_reward = scores[0] if isinstance(scores, (list, tuple)) else scores
            if isinstance(raw_reward, (list, tuple)) and len(raw_reward) > 0:
                raw_reward = raw_reward[0]
            try:
                reward = float(raw_reward)
            except Exception:
                logger.warning(
                    f"Unexpected reward format: {raw_reward} "
                    f"(type={type(raw_reward)}) - setting reward=0.0"
                )
                reward = 0.0

            done = dones[0] if isinstance(dones, list) else dones
            current_info = infos[0] if isinstance(infos, list) else infos or {}

            admissible_commands = _extract_admissible_commands(current_info)

            return {
                "observation": str(current_obs),
                "reward": reward,
                "done": bool(done),
                "info": {
                    "admissible_commands": admissible_commands,
                    "won": current_info.get("won", False),
                    "lost": current_info.get("lost", False),
                },
            }

        except Exception as e:
            logger.error(f"Step failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/close")
async def close_env():
    """Tear down the active env. Call this when a rollout is finished so the
    underlying TextWorld subprocess is not kept alive between episodes.

    Does not evict cached split envs unless `drop_cache=true` is ever added.
    """
    global current_env, current_obs, current_info, current_env_key

    async with env_lock:
        _dispose_current_env()
        current_obs = None
        current_info = None
        current_env_key = None
        return {"status": "closed"}


@app.get("/admissible_commands")
async def get_admissible_commands():
    if current_info is None:
        return {"commands": []}
    return {"commands": _extract_admissible_commands(current_info)}


@app.get("/info")
async def get_info():
    if current_info is None:
        return {"info": {}}

    return {
        "info": {
            "task": current_info.get(
                "extra.gamefile", current_info.get("task", "unknown")
            ),
            "goal": current_info.get("goal", current_info.get("task_description", "")),
            "won": current_info.get("won", False),
            "lost": current_info.get("lost", False),
            "admissible_commands_count": len(
                current_info.get("admissible_commands", [])
            ),
        }
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)