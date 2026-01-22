from fastapi import FastAPI
from scienceworld import ScienceWorldEnv

app = FastAPI()

env = ScienceWorldEnv("", serverPath=None, envStepLimit=100)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/reset")
def reset():
    observation = env.reset()
    return {"observation": observation}


@app.get("/load")
def load(task_name: str, variation_idx: int):
    env.load(task_name, variation_idx)
    observation = env.reset()
    return {"observation": observation}


@app.get("/step")
def step(action: str):
    observation, reward, done, info = env.step(action)
    return {"observation": observation, "reward": reward, "done": done, "info": info}
