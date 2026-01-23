# ALFWorld Environment

## Overview

ALFWorld (Action Learning From World) is an interactive text-based environment that simulates a household setting. It is built upon the [TextWorld](https://www.microsoft.com/en-us/research/project/textworld/) framework and is designed to test an agent's ability to perform complex, multi-step tasks by following natural language instructions.

## Components

- [Environment](environment.md) - ALFWorldEnv class reference
- [Tools](tools.md) - ALFWorld interaction tools
- [Rewards](rewards.md) - ALFWorld reward functions

## Quick Start

For most use cases, you can use the ALFWorldEnv class directly:

```python
from agentfly.envs.alfworld_env import ALFWorldEnv

# Create and start the environment
env = ALFWorldEnv()
await env.start()

# Reset to start a new episode
obs, info = await env.reset()

# Take actions
obs, reward, done, info = await env.step("go to kitchen")
```

For HTTP-based integration, see the [HTTP Server](http_server.md) documentation.

## Key Features

* **Docker-based Isolation**: Runs in containers for consistent environments
* **Task Selection**: Support for both random and specific task selection
* **HTTP API**: RESTful interface for language-agnostic integration
* **Comprehensive State Access**: Full access to observations, admissible commands, and environment state
* **Automatic Recovery**: Built-in error handling and container restart capabilities
