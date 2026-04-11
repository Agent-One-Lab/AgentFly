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

## Security Features

The environment implements multiple security layers:

* **Container Isolation**: Each environment runs in a separate Docker container
* **Read-only Filesystem**: Container filesystem is mounted read-only
* **Capability Dropping**: All Linux capabilities are dropped (``cap_drop=["ALL"]``)
* **Process Limits**: Limited to 256 processes (``pids_limit=256``)
* **Resource Limits**: CPU and memory usage are strictly controlled
* **Network Isolation**: Containers use isolated bridge networks
* **Timeout Protection**: Execution timeouts prevent infinite loops

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

### State persistence within sessions:

```python
# Variables persist between steps
await env.step("x = [1, 2, 3]")
await env.step("x.append(4)")
result = await env.step("print(x)")
# Output: [1, 2, 3, 4]

# Reset clears state
await env.reset()
result = await env.step("print(x)")
# Output: NameError: name 'x' is not defined
```
