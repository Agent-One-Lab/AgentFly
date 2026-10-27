# Code Environment

The Code Environment runs Python snippets in a container (enroot by default) with CPU, memory and time limits. It keeps model-written code off your own process and filesystem, but it is not a security boundary: see [Limits and isolation](index.md#limits-and-isolation).

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

Errors in the code don't raise: `step` returns the error text as its result, and
`code_interpreter` passes it to the model as the tool result.

- **Python error**: the traceback (stderr).
- **Timeout** (10 s wall time): the server kills the snippet's process and returns `"Execution timed out"`.
- **CPU or memory limit**: `"Process killed (signal N): CPU time or memory limit exceeded"`.

```python
result = await env.step("while True: pass")
# result == "Execution timed out" (no exception; the next step works normally)
```

`step` raises only if the sandbox isn't started or the server sends an unexpected response.
The container is not restarted by `step`. When the pool reuses a sandbox it calls `reset()`,
which restarts the container if it doesn't answer within 20 s or the connection fails.

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
