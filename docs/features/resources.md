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

## Resource Types

AgentFly currently exposes four spec dataclasses from `agentfly.resources`:

- **`ContainerResourceSpec`** — Enroot-backed containers, including pre-built environment images. Tools/rewards request these to get an isolated runtime they can `step()` or `run_cmd()` against. The `category` field discriminates between generic containers (`"container"`) and built-in environment categories (`"python_env"`, `"alfworld"`, `"webshop"`, `"scienceworld"`). Configurable fields include `image`, `cpu_count`, `mem_limit`, `gpus`, `ports`, `mount`, etc.
- **`VLLMModelResourceSpec`** — Locally launched vLLM model service. Used when a tool/reward needs to call out to a model served from the same cluster (e.g., a separate judge model). Fields include `model_name_or_path`, `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`, `gpu_memory_utilization`, `template`.
- **`APIModelResourceSpec`** — A pre-existing OpenAI-compatible HTTP endpoint. No process is launched; the engine just records connection details (`base_url`, `host`, `api_key`, `request_timeout`). Use this when calling an external service (OpenAI, Anthropic, in-house API) from a tool or reward.
- **`BaseResourceSpec`** — Base class. You normally use one of the three above; subclass `BaseResourceSpec` only when adding a new resource kind.

Predefined environment specs (built on `ContainerResourceSpec`) are exported by the corresponding env modules:

- `PythonSandboxSpec` — `agentfly.envs.python_env`
- `ALFWorldSpec` — `agentfly.envs.alfworld_env`
- `WebShopSpec` — `agentfly.envs.webshop_text_env`
- `ScienceWorldSpec` — `agentfly.envs.scienceworld_env`

These are the typical specs you pass to `context.acquire_resource(spec=...)` from tools and rewards.

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

## Examples

The following are the same shape: a tool or reward declares a `Context` parameter, calls `context.acquire_resource(spec=..., scope=..., backend=...)`, and uses the returned resource via its `step` (or `run_cmd`) interface.

### Tool: acquiring a Python sandbox

`code_interpreter` is a stateful tool that runs code in a `PythonSandboxEnv`. It acquires the sandbox per rollout and lets `ResourceEngine` handle pooling and lifecycle:

```python
--8<-- "src/agentfly/tools/src/code/tools.py:code_interpreter_example"
```

Notes:

- The tool takes `context: Context` as its second parameter; AgentFly injects this automatically during rollouts.
- Same `(spec, scope)` pair within the same rollout returns the **same** resource instance, so subsequent calls see persistent variable state inside the sandbox.

### Reward: sharing the rollout's sandbox

A reward can acquire the same resource a tool uses, by passing the same `spec` and `scope`. `code_reward_test` re-runs the agent's prediction in the same sandbox the agent already used:

```python
--8<-- "src/agentfly/rewards/code_reward.py:code_reward_test_example"
```

Both `code_interpreter` (tool) and `code_reward_test` (reward) call `context.acquire_resource(spec=PythonSandboxSpec, scope="global", backend="local")`. Because the spec and scope match, the engine hands them the **same** running container. No explicit handle is passed between them.

### Reward: acquiring an environment container

`webshop_reward` queries the WebShop environment's reward via the same resource the WebShop tools use:

```python
--8<-- "src/agentfly/rewards/webshop_reward.py:webshop_reward_example"
```

Same pattern: `acquire_resource(spec=WebShopSpec, ...)` → `await env.step(...)`. The reward never needs to know how the tools acquired or set up the env.

### Tool: dynamic image (built at call time)

When the image you need depends on per-rollout metadata (e.g., a SWE-bench task ships its own container image), build the `ContainerResourceSpec` at call time instead of using a module-level constant:

```python
from agentfly.resources import ContainerResourceSpec

image_id = context.metadata.get("image_id")  # e.g. set by the dataset loader
spec = ContainerResourceSpec(category="container", image=image_id)
container = await context.acquire_resource(
    id=context.rollout_id,
    spec=spec,
    backend=context.resource_backend,
)
output = await container.run_cmd("ls /workspace")
```

The shell tool in `agentfly.tools.src.shell.tools` uses this pattern to pick up arbitrary container images from per-rollout metadata.

## Migration Guidance

When updating existing code:

- Replace direct environment-manager usage with `context.acquire_resource(...)`.
- Keep tools/rewards stateless with respect to resource handles; reacquire by spec each call.
- Use the same spec and scope across tools/rewards when they must share state.
- Let the engine manage pooling and cleanup.
