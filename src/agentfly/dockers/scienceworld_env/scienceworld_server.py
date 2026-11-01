import os

from fastapi import FastAPI
from scienceworld import ScienceWorldEnv

app = FastAPI()

# Env actions per episode before ScienceWorld marks it done. Subagents share one env, so
# their episodes can need more than the single-agent default.
STEP_LIMIT = int(os.environ.get("SCIENCEWORLD_STEP_LIMIT", "100"))

env = ScienceWorldEnv("", serverPath=None, envStepLimit=STEP_LIMIT)


def observation_of(reset_result):
    """The observation text of ``env.reset()``: ScienceWorld 1.2+ returns
    ``(observation, info)``, older versions the observation alone."""
    return reset_result[0] if isinstance(reset_result, tuple) else reset_result


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/reset")
def reset():
    observation = observation_of(env.reset())
    return {"observation": observation}


@app.get("/load")
def load(task_name: str, variation_idx: int):
    env.load(task_name, variation_idx)
    observation = observation_of(env.reset())
    return {"observation": observation}


@app.get("/step")
def step(action: str):
    observation, reward, done, info = env.step(action)
    return {"observation": observation, "reward": reward, "done": done, "info": info}
