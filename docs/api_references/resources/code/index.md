# Code Environment

## Overview

The Code Environment runs Python snippets in a container (the `reasonwang/python-env` image, started with enroot by default) and returns their output. Each container runs a small FastAPI server; every snippet runs in a fresh Python process with time, CPU and memory limits.

The container keeps model-written code off your own Python process and filesystem, but it is not a security boundary: see [Limits and isolation](#limits-and-isolation).

## Components

- [Environment](environment.md) - PythonSandboxEnv class reference
- [HTTP Server](http_server.md) - FastAPI server implementation
- [Tools](tools.md) - Code execution tools
- [Rewards](rewards.md) - Code evaluation rewards

## Quick Start

For most use cases, you interact with the code sandbox through `Context` and the resource engine (this is how tools/rewards acquire environments during rollouts):

```python
from agentfly.core import Context
from agentfly.envs.python_env import PythonSandboxSpec

context = Context(rollout_id="demo")
env = await context.acquire_resource(
    spec=PythonSandboxSpec,
    scope="rollout",
    backend="local",
)

# Execute Python code
result = await env.step("print('Hello, World!')")
print(result)  # Output: Hello, World!

# Each call is a fresh Python process: define everything a snippet needs in that snippet
await env.step("x = 42")
result = await env.step("print(x)")
print(result)  # NameError: name 'x' is not defined

# Release/kill the rollout-scoped resource
await context.end_resource(scope="rollout")
```

For HTTP-based integration, see the [HTTP Server](http_server.md) documentation.

## Key Features

* **Fresh process per call**: each snippet runs in a new Python process, so nothing carries over between calls. This is deliberate: `code_interpreter` takes sandboxes from a pool shared by all rollouts, and leftover state would leak from one rollout into the next.
* **Normal script semantics**: functions see top-level names and imports, and `if __name__ == "__main__":` blocks run.
* **Limits per call**: 10 s wall time, 8 CPU-seconds, 1 GiB memory; output is truncated to 16 KB.
* **Blocked modules**: `subprocess`, `multiprocessing`, `ctypes`, `cffi`, `sys`, `signal`, `resource`, `ssl`, `pty`, `fcntl`, `socketserver` and `importlib.machinery` cannot be imported.
* **HTTP API**: `POST /exec` and `GET /health` (see [HTTP Server](http_server.md)).

## Limits and isolation

With the default enroot engine, each sandbox has its own filesystem (the image's root filesystem), but it is **not** a security boundary: the network is open, processes run as your own user, and the host's process list is visible. Only run it on machines where executing model-written code with your user's permissions is acceptable. With `AF_CONTAINER_ENGINE=docker`, Docker's defaults add a separate process list and network namespace; AgentFly adds no further hardening.
