# Environments

We define an **environment** as a *resource-backed execution context* that agents can interact with via tools and rewards. In the current design, most environments are implemented as:

- A concrete resource class (e.g. `PythonSandboxEnv`, `ALFWorldEnv`, `WebAgentTextEnv`, `ScienceWorldEnv`) that subclasses `ContainerResource` or a similar base.
- A corresponding `ResourceSpec` (e.g. `PythonSandboxSpec`, `ALFWorldSpec`, `WebShopSpec`, `ScienceWorldSpec`) that describes how to start and scale that resource (image, ports, limits, etc.).

Environments play two main roles:

1. **Isolation**: Operations like executing code or running simulators are done inside containers or dedicated processes, so different tasks and rollouts do not interfere with each other.
2. **Scaling**: The `ResourceEngine` can maintain multiple instances of the same environment, controlled by fields such as `max_global_num` in the `ResourceSpec` and engine configuration.

## Resource Management System

Tools and rewards no longer talk to an `EnvironmentManager` directly. Instead, they use the **rollout `Context`** to acquire and manage resources:

- `context.acquire_resource(spec=..., scope="rollout" | "global", backend="local")` returns a resource instance (e.g. `PythonSandboxEnv`, `ALFWorldEnv`, `WebAgentTextEnv`, `ScienceWorldEnv`).
- The `Context` keeps track of which specs have been acquired for the current rollout and whether this is the first acquisition (via `is_spec_acquired` and `last_acquire_was_first`), so tools can decide when to call `reset`.
- The underlying **`ResourceEngine`** handles pooling, starting, resetting, and ending resources across backends (local, Slurm, etc.).

Sharing works by spec and scope:

- Tools and rewards that call `context.acquire_resource` with the **same `ResourceSpec` and scope** in the same rollout will share the same environment instance.
- Different rollouts get their own instances automatically; you do not need to pass explicit ids in the tool or reward definitions.

## Definition

Environment classes remain responsible for the *interaction protocol* with the underlying system. Typically, they implement:

- `start`: (asynchronous) Start the resource (e.g. container, local process) and connect any necessary clients (HTTP, sockets, etc.).
- `reset`: (asynchronous) Reset the environment state (e.g. load a new task, clear globals) without restarting the whole resource when possible.
- `step`: Main interface to interact with the environment (e.g. for code interpreter, execute the code and return results; for ALFWorld/WebShop/ScienceWorld, perform an action and return observations/rewards).
- `end` / `close`: (asynchronous) Cleanly terminate or close the resource when no longer needed.

From the perspective of tools and rewards, environments are accessed **only via `Context` and `ResourceSpec`**; the details of pooling and lifecycle are abstracted away by the `ResourceEngine`.
