# Code Environment

The Code Environment provides a secure Python sandbox execution environment using Docker containers. It enables safe execution of untrusted Python code with strict resource limits and security controls.

## PythonSandboxEnv Class Reference

::: agentfly.envs.python_env.PythonSandboxEnv
    options:
      members: true
      show_inheritance: true
      show_source: true

## Configuration Parameters

The PythonSandboxEnv supports the following configuration options:

* **image** (str): Docker image to use (default: ``reasonwang/python-http-env:latest``)
* **runtime** (str): Container runtime (default: ``runc``)
* **cpu** (int): CPU limit in cores (default: ``2``)
* **mem** (str): Memory limit (default: ``2g``)
* **start_timeout** (float): Timeout for container startup in seconds (default: ``60.0``)
* **max_episodes** (int): Maximum number of episodes per container (default: ``100``)
* **host_ip** (str): Host IP for binding (default: ``127.0.0.1``)
* **container_port** (int): Container port for HTTP server (default: ``8000``)

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

### Basic usage with direct instantiation:

```python
from agentfly.envs.python_env import PythonSandboxEnv

# Create environment with custom settings
env = PythonSandboxEnv(
    cpu=1,
    mem="1g",
    start_timeout=30.0
)

# Start the container
await env.start()

# Execute code
result = await env.step("import math; print(math.pi)")
# Output: 3.141592653589793

# Clean up
await env.aclose()
```

### Using the factory method for pool management:

```python
# Use factory method for environment pools
env = await PythonSandboxEnv.acquire()

# Environment is pre-started and reset
result = await env.step("print('Ready to use!')")

await env.aclose()
```

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
