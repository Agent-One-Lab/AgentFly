# ALFWorld Environment

`ALFWorldEnv` is a managed resource created by `ResourceEngine` runners. In AgentFly, tools and rewards acquire it via `Context.acquire_resource(spec=ALFWorldSpec, ...)`.

## Class Reference

::: agentfly.envs.alfworld_env.ALFWorldEnv
    options:
      members: true
      show_inheritance: true
      show_source: true

## Usage Examples

### Basic Usage

```python
from agentfly.core import Context
from agentfly.envs.alfworld_env import ALFWorldSpec

context = Context(rollout_id="demo")
env = await context.acquire_resource(
    spec=ALFWorldSpec,
    scope="rollout",
    backend="local",
)

# Reset to start a new episode
obs, info = await env.reset()
print(f"Initial observation: {obs}")

# Take an action
obs, reward, done, info = await env.step("go to kitchen")
print(f"Reward: {reward}, Done: {done}")

# Cleanup (kill rollout-scoped resource)
await context.end_resource(scope="rollout")
```

### Custom Configuration

```python
# NOTE: Acquire `ALFWorldEnv` via `Context`/`ResourceEngine` (not via `ALFWorldEnv(...)` constructor).
# Most task customization is done via `await env.reset(env_args=...)` once the managed environment is acquired.
# Recommended example (Context-based)
from agentfly.core import Context
from agentfly.envs.alfworld_env import ALFWorldSpec

context = Context(rollout_id="demo")
env = await context.acquire_resource(
    spec=ALFWorldSpec,
    scope="rollout",
    backend="local",
)

# Reset to a specific task
obs, info = await env.reset(
    env_args={"task_id": "trial_T20190907_212755_456877"}
)

# Cleanup (kill rollout-scoped resource)
await context.end_resource(scope="rollout")

# --- Legacy snippet below (deprecated) ---
env = ALFWorldEnv(
    image="custom/alfworld-env:latest",
    cpu=4,
    mem="4g",
    train_eval="valid_seen",
    max_episodes=100
)
await env.start()

# Reset to a specific task
obs, info = await env.reset(
    env_args={"task_id": "trial_T20190907_212755_456877"}
)
```

## Configuration parameters (via `ALFWorldSpec`)

In AgentFly, `ALFWorldEnv` is configured through `ALFWorldSpec` (a `ResourceSpec`) and acquired via `Context.acquire_resource(...)`.

- Container image/ports/concurrency are encoded in `ALFWorldSpec` (including `ALFWorldSpec.max_global_num`).
- For per-episode task selection, call `await env.reset(env_args=...)` after acquiring the environment.
