# Code Environment

The Code Environment provides a secure Python sandbox execution environment using Docker containers. It enables safe execution of untrusted Python code with strict resource limits and security controls.

## PythonSandboxEnv Class Reference

::: agentfly.envs.python_env.PythonSandboxEnv
    options:
      members: true
      show_inheritance: true
      show_source: true

## ResourceSpec configuration

In AgentFly, `PythonSandboxEnv` is created and started by `ResourceEngine` runners using a `ResourceSpec`.

Most users should not instantiate `PythonSandboxEnv` directly. Instead, acquire it via:

- `Context.acquire_resource(spec=PythonSandboxSpec, scope=..., backend=...)`
- tools/rewards during rollouts (recommended)

## Limits

* **Own filesystem**: each environment runs in its own container (enroot by default)
* **Per-call limits**: 10 s wall time, 8 CPU-seconds, 1 GiB memory
* **Blocked modules**: `subprocess`, `multiprocessing`, `ctypes` and other process/system modules
* **Fresh process per call**: no state carries over between `step` calls

It is not a security boundary (open network, runs as your user): see [Limits and isolation](index.md#limits-and-isolation).

## Usage Examples

### Usage via Context

```python
from agentfly.core import Context
from agentfly.envs.python_env import PythonSandboxSpec

# Acquire the sandbox
env = await context.acquire_resource(
    spec=PythonSandboxSpec,
    scope="rollout",
    backend="local",
)

# env is already started when acquired from ResourceEngine

# Execute code
result = await env.step("import math; print(math.pi)")
# Output: 3.141592653589793

# Cleanup
await context.end_resource(scope="rollout")
```

### Pooling and reuse

AgentFly handles pooling/reuse via `ResourceEngine`. Sandboxes are acquired via `Context.acquire_resource(spec=PythonSandboxSpec, ...)`, and the maximum number of concurrently existing sandboxes is controlled by `PythonSandboxSpec.max_global_num`.

## Error Handling and Recovery

The environment includes automatic error handling:

```python
# Automatic container restart on timeout
try:
    result = await env.step("while True: pass")  # Infinite loop
except Exception:
    # Environment automatically restarts container
    result = await env.step("print('Recovered!')")
```

### No state between steps:

```python
# Each step is a fresh Python process
await env.step("x = [1, 2, 3]")
result = await env.step("print(x)")
# Output: NameError: name 'x' is not defined

# Put everything a computation needs in one step
result = await env.step("x = [1, 2, 3]\nx.append(4)\nprint(x)")
# Output: [1, 2, 3, 4]
```
