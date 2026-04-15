# Resources

AgentFly has migrated from environment-centric management to a **unified resource system**.
In this system, environments are one type of resource alongside other runtime units
(for example, model engines). Tools and rewards should depend on `Context` and
`ResourceSpec`, not on a legacy environment manager.

## Core Concepts

- **Resource**: A runtime unit that can be started, reused, reset, and released
  (for example `PythonSandboxEnv`, `ALFWorldEnv`, `WebAgentTextEnv`, `ScienceWorldEnv`).
- **ResourceSpec**: A declarative spec that describes how to provision and scale a resource
  (for example image, backend, limits such as `max_global_num`).
- **Context**: The rollout-scoped interface used by tools and rewards to acquire resources.
- **ResourceEngine**: The backend engine that handles pooling, lifecycle, and isolation
  across local and cluster backends.

## Resource Acquisition

Tools and rewards acquire resources through `Context`:

- `context.acquire_resource(spec=..., scope="rollout" | "global", backend="local")`
- Returned value is a live resource instance implementing the task interface
  (usually `step`, plus lifecycle methods).

The `Context` also tracks acquisition state for the rollout (for example whether a
given spec was already acquired), which allows callers to perform one-time reset logic
without manually managing resource ids.

## Sharing and Isolation

Sharing behavior is determined by **spec + scope**:

- Same rollout + same `ResourceSpec` + same scope -> shared resource instance.
- Different rollouts -> isolated resource instances by default.
- Pool size and scaling are configured in the spec/engine, not in tool business logic.

This gives you deterministic sharing where needed and clean isolation across concurrent rollouts.

## Lifecycle Responsibilities

Resource classes are responsible for the protocol to the underlying runtime and should
normally expose asynchronous lifecycle methods:

- `start`: initialize container/process and clients.
- `reset`: reset task state when reuse is preferred over restart.
- `step`: main interaction entry point.
- `end` / `close`: terminate and release runtime state.

The `ResourceEngine` orchestrates these lifecycle operations; tools and rewards should
focus on task logic instead of manual resource bookkeeping.

## Container Resource

`ContainerResource` is the standard runtime wrapper for Docker/enroot-backed workloads
in AgentFly. It implements the `BaseResource` contract and exposes container-oriented
operations (for example command execution via `run_cmd`) while still fitting into the
same acquire/release lifecycle managed by the engine.

Typical usage is still through `Context.acquire_resource(...)` in tools/rewards; callers
should not instantiate container resources directly.

For API details, see:

- `api_references/resources/resources.md` (`ResourceSpec`, `BaseResource`, `ContainerResource`)
- `api_references/resources/resource_engine.md` (`ResourceEngine`)

## Migration Guidance

When updating existing code:

- Replace direct environment-manager usage with `context.acquire_resource(...)`.
- Keep tools/rewards stateless with respect to resource handles; reacquire by spec each call.
- Use the same spec and scope across tools/rewards when they must share state.
- Let the engine manage pooling and cleanup.
