"""
ALFWorld HTTP Server

This script launches a FastAPI web server to provide HTTP endpoints for interacting
with an underlying ALFWorld environment. It is designed to be run within a
Docker container.

Key features include:
- Endpoint for resetting the environment, with support for selecting a specific
  task by its unique `task_id`.
- Endpoints for stepping through an episode, and retrieving state information
  like admissible commands.
- On-startup scanning of ALFWorld data files to build a cache of available
  tasks, enabling the task selection feature.
"""

import sys
import yaml
import glob
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil      
from copy import deepcopy   
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# Import ALFWorld
try:
    import alfworld
    import alfworld.agents.environment as environment
    import alfworld.agents.modules.generic as generic
    from alfworld.agents.environment import get_environment
except ImportError as e:
    logger.error(f"Failed to import ALFWorld: {e}")
    sys.exit(1)

app = FastAPI()

# --- Global variables for environment management ---
# Preserving the original structure
config: Optional[Dict] = None
current_env: Optional[Any] = None
current_obs: Optional[str] = None
current_info: Optional[Dict] = None

# New globals required for task selection functionality
task_cache: Dict[str, Dict[str, Dict]] = {}
split_envs: Dict[str, Any] = {} # Caches environments for random sampling per split


# --- Pydantic Models ---

class ActionRequest(BaseModel):
    action: str

# MODIFIED: Updated ResetRequest to include split and optional task_id
class ResetRequest(BaseModel):
    split: str = Field(
        "train",
        description="The data split to use ('train', 'valid_seen', 'valid_unseen').",
    )
    task_id: Optional[str] = Field(
        None, description="Optional: A specific task_id to load."
    )


# --- NEW: Helper functions for task management ---

def _create_env(split: str, task_info: Optional[Dict] = None):
    """
    Creates and initializes an AlfredTWEnv instance.

    This function supports two modes:
    1.  **Specific Task Mode:** If `task_info` is provided, it modifies the global
        config on a deep copy to point directly to the specific trial directory.
        This ensures the environment loads only that single game.
    2.  **Random Task Mode:** If `task_info` is None, it creates a standard
        environment that will sample tasks randomly from the specified `split`.

    :param split: The data split ('train', 'eval_in_distribution', etc.).
    :type split: str
    :param task_info: A dictionary containing metadata for a specific task,
                      including its 'game_file' path.
    :type task_info: Optional[Dict]
    :return: An initialized AlfredTWEnv instance.
    :rtype: Any
    """
    global config

    env_type = config.get("env", {}).get("type", "AlfredTWEnv")
    EnvClass = get_environment(env_type)

    # ---- CASE 1: load one specific task ---------------------------------
    if task_info:
        trial_dir = os.path.dirname(task_info["game_file"])        # …/trial_Txxxx_yyyy/
        task_type_dir = os.path.basename(os.path.dirname(trial_dir))
        task_type_slug = task_type_dir.split("-")[0]               # e.g. pick_heat_then_place_in_recep

        # integer id 1-6 that Alfworld expects in env.task_types
        TASK_SLUG2ID = {
            "pick_and_place_simple": 1,
            "look_at_obj_in_light": 2,
            "pick_clean_then_place_in_recep": 3,
            "pick_heat_then_place_in_recep": 4,
            "pick_cool_then_place_in_recep": 5,
            "pick_two_obj_and_place": 6,
        }
        task_type_id = TASK_SLUG2ID[task_type_slug]

        # deepcopy so we never mutate the global template
        cfg = deepcopy(config)
        
        split_to_path_key = {
            "train": "train_data_path",
            "valid_seen": "valid_seen_data_path",
            "valid_unseen": "valid_unseen_data_path",
        }
        if "dataset" not in cfg:
            cfg["dataset"] = {}
        
        if "env" not in cfg:
            cfg["env"] = {}
        
        cfg["dataset"]["data_path"] = trial_dir            # look at only this game
        cfg["dataset"]["num_train_games"] = 1              # safety cap
        cfg["dataset"][split_to_path_key[split]] = trial_dir

        # Env section
        cfg["env"]["task_types"] = [task_type_id]          # e.g. [4] for pick_heat_then_place_in_recep

        logger.info(f"Loading single task {task_info['task_id']}")
        logger.info(f"data_path set to {trial_dir} (split={split})")

        # NOTE: AlfredTWEnv expects 'train', 'eval_in_distribution', or 'eval_out_of_distribution'.
        # Using 'train' is the safest option for a single-task env because it guarantees
        # the variable `data_path` is set inside `collect_game_files`.
        env = EnvClass(cfg, train_eval="train")
    # ---- CASE 2: random as before ---------------------------------------
    else:
        env = EnvClass(config, train_eval=split)

    env = env.init_env(batch_size=1)
    return env


def _scan_and_cache_tasks():
    """
    Scans the ALFWorld data directory to find all task files and cache their metadata.

    This function traverses the directory structure defined by ALFWorld (e.g.,
    `.../json_2.1.1/train/pick_and_place_simple/trial_.../`) to identify all
    individual task trials. It stores their metadata (task_id, type, path) in the
    global `task_cache` dictionary, keyed by data split. This cache is used
    to power the task selection functionality.
    """
    global task_cache, config
    
    # Get the data root from config or environment
    data_path = os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")
    data_path = os.path.expanduser(data_path)
    
    # Initialize task dictionary
    task_cache = {
        "train": {},
        "valid_seen": {},
        "valid_unseen": {},
        "test_unseen": {}
    }
    
    logger.info(f"Scanning for tasks in base path: {data_path}/json_2.1.1/train")
    
    # Scan for tasks in each split
    for split in task_cache.keys():
        split_path = os.path.join(data_path, "json_2.1.1", split)
        if not os.path.exists(split_path):
            logger.warning(f"Split directory not found: {split_path}")
            continue
            
        # First level: task type directories
        for task_type_dir in os.listdir(split_path):
            task_type_path = os.path.join(split_path, task_type_dir)
            if not os.path.isdir(task_type_path):
                continue
                
            # The directory name contains the task type and parameters
            task_type = task_type_dir.split('-')[0]
            
            # Second level: trial directories
            for trial_dir in os.listdir(task_type_path):
                if not trial_dir.startswith('trial_'):
                    continue
                    
                trial_path = os.path.join(task_type_path, trial_dir)
                if not os.path.isdir(trial_path):
                    continue
                    
                # The trial directory name is the task_id
                task_id = trial_dir
                
                # Check if traj_data.json exists
                traj_data_path = os.path.join(trial_path, "traj_data.json")
                if os.path.exists(traj_data_path):
                    try:
                        task_cache[split][task_id] = {
                            "task_id": task_id,
                            "task_type": task_type,
                            "game_file": os.path.join(trial_path, "game.tw-pddl")
                        }
                    except Exception as e:
                        logger.warning(f"Could not process task directory {trial_path}: {e}")
    
    counts = {k: len(v) for k, v in task_cache.items()}
    logger.info(f"Task cache created. Found tasks: {counts}")

# --- FastAPI Event Handlers and Endpoints ---

@app.on_event("startup")
async def startup_event():
    """
    Handles server startup logic.

    This function is executed once when the FastAPI application starts. It loads
    the ALFWorld configuration file and then triggers the task scanning and
    caching process.
    """
    global config
    
    config_path = os.environ.get("ALFWORLD_CONFIG", "/srv/base_config.yaml")
    
    try:
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], config_path]
        try:
            config = generic.load_config()
            logger.info("ALFWorld configuration loaded successfully")
            # NEW: Scan for tasks now that config is loaded
            _scan_and_cache_tasks()
        finally:
            sys.argv = original_argv
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

@app.get("/health")
async def health():
    """Provides a simple health check endpoint to confirm the server is running."""
    return {"status": "ok", "service": "alfworld"}

#Endpoint to see available tasks, essential for using the new feature
@app.get("/available_tasks")
async def get_available_tasks(split: str = "train"):
    """
    Returns a list of all available tasks for a given data split.

    :param split: The data split ('train', 'valid_seen', 'valid_unseen').
    :return: A JSON object containing a list of task metadata dictionaries.
    :raises HTTPException: 404 if the split is invalid.
    """
    if split not in task_cache:
        raise HTTPException(status_code=404, detail=f"Invalid split: {split}")
    return {"tasks": list(task_cache[split].values())}


@app.post("/reset")
async def reset(request: ResetRequest):
    """
    Resets the environment to start a new episode.

    Can be configured to load a specific task by its `task_id` or a random
    task from the specified `split`.

    :param request: A `ResetRequest` object containing the `split` and optional `task_id`.
    :return: A JSON object with the initial observation and environment info.
    :raises HTTPException: 400 for invalid split, 404 if `task_id` not found,
                           500 for internal errors during reset.
    """
    global current_env, current_obs, current_info, config
    
    try:
        split = request.split
        task_id = request.task_id
        
        if split not in task_cache:
            raise HTTPException(status_code=400, detail=f"Invalid split: {split}")
        
        # If task_id is provided, validate and use it
        task_info = None
        if task_id:
            if task_id not in task_cache[split]:
                raise HTTPException(
                    status_code=404,
                    detail=f"Task ID '{task_id}' not found in split '{split}'."
                )
            task_info = task_cache[split][task_id]
        
        # Create environment with or without specific task
        current_env = _create_env(split, task_info)
        
        # Reset the environment
        obs, info = current_env.reset()
        logger.info("Environment reset completed")
        
        current_obs = obs[0] if isinstance(obs, list) else obs
        current_info = info[0] if isinstance(info, list) else info or {}
        
        admissible_commands = []
        if current_info and 'admissible_commands' in current_info:
            cmds = current_info['admissible_commands']
            admissible_commands = cmds[0] if isinstance(cmds, list) and len(cmds) > 0 and isinstance(cmds[0], list) else cmds
        
        goal = ''
        if current_info:
            goal = (current_info.get('goal') or 
                   current_info.get('task_description') or
                   current_info.get('task_desc') or
                   current_info.get('description') or '')
        
        task = current_info.get('extra.gamefile', current_info.get('task', 'unknown')) if current_info else 'unknown'
        
        logger.info(f"Reset successful - Task: {task}, Goal: {goal}")
        
        return {
            "observation": str(current_obs),
            "info": {
                "admissible_commands": admissible_commands,
                "task": task,
                "goal": goal,
                "steps_taken": 0,
            }
        }
        
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(request: ActionRequest):
    """
    Executes a single action in the environment.

    :param request: An `ActionRequest` object containing the action string.
    :return: A JSON object with the new observation, reward, done status, and info.
    :raises HTTPException: 400 if env not initialized, 500 for internal step errors.
    """
    global current_env, current_obs, current_info
    
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        actions = [request.action]
        obs, scores, dones, infos = current_env.step(actions)
        
        current_obs = obs[0] if isinstance(obs, list) else obs
        
        # `scores` can be a list, numpy array, or even a tuple coming from TextWorld.
        raw_reward = scores[0] if isinstance(scores, (list, tuple)) else scores

        # If we still have an iterable (e.g., (0.0,)), grab the first element.
        if isinstance(raw_reward, (list, tuple)) and len(raw_reward) > 0:
            raw_reward = raw_reward[0]

        # Cast to float defensively
        try:
            reward = float(raw_reward)
        except Exception:
            # Fallback: zero reward on unexpected format and log the error.
            logger.warning(f"Unexpected reward format: {raw_reward} (type={type(raw_reward)}) — setting reward=0.0")
            reward = 0.0

        done = dones[0] if isinstance(dones, list) else dones
        current_info = infos[0] if isinstance(infos, list) else infos or {}
        
        admissible_commands = []
        if current_info and 'admissible_commands' in current_info:
            cmds = current_info['admissible_commands']
            admissible_commands = cmds[0] if isinstance(cmds, list) and len(cmds) > 0 and isinstance(cmds[0], list) else cmds
        
        return {
            "observation": str(current_obs),
            "reward": reward,
            "done": bool(done),
            "info": {
                "admissible_commands": admissible_commands,
                "won": current_info.get('won', False),
                "lost": current_info.get('lost', False),
            }
        }
        
    except Exception as e:
        logger.error(f"Step failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admissible_commands")
async def get_admissible_commands():
    """Get the current list of admissible commands."""
    if current_info is None:
        return {"commands": []}
    
    admissible_commands = []
    if 'admissible_commands' in current_info:
        cmds = current_info['admissible_commands']
        admissible_commands = cmds[0] if isinstance(cmds, list) and len(cmds) > 0 and isinstance(cmds[0], list) else cmds
    
    return {"commands": admissible_commands}

@app.get("/info")
async def get_info():
    """Get current environment information."""
    if current_info is None:
        return {"info": {}}
    
    return {
        "info": {
            "task": current_info.get('extra.gamefile', current_info.get('task', 'unknown')),
            "goal": current_info.get('goal', current_info.get('task_description', '')),
            "won": current_info.get('won', False),
            "lost": current_info.get('lost', False),
            "admissible_commands_count": len(current_info.get('admissible_commands', [])),
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 