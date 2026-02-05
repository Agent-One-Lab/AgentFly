# Resource Engine Design Proposal

This document describes a **decoupled resource engine** that generalizes the current `EnvironmentManager` to manage multiple resource types (vLLM deployed models, containers) across multiple backends (local, Slurm, AWS, K8s). Tools and rewards communicate with the engine to acquire, use, and release resources.

---

## 1. Goals

- **Decouple resource type from execution backend**: Same resource kind (e.g. container) can run on local (enroot), Slurm, AWS, or K8s.
- **Unified API for tools/rewards**: Single entry point to start, acquire, monitor, control, and call resources.
- **Dynamic scaling**: Open or close resources based on request frequency or current utilization.
- **Multiple invocation modes**: Call resources via **MCP tool-call** (MCP server) or **input-text** (e.g. HTTP, stdin).

---

## 2. System Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Tools / Rewards (consumers)                        │
│  acquire(id) → use resource (mcp-tool-call / input-text) → release(id)   │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ResourceEngine (facade)                           │
│  start / acquire / release / reset / monitor / control / aclose          │
│  + dynamic scaling policy (scale up/down by request rate or capacity)    │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  LocalRunner    │       │  SlurmRunner     │       │  CloudRunner     │
│  (enroot,       │       │  (sbatch/srun)   │       │  (AWS / K8s)     │
│   local procs)  │       │                  │       │                  │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Resource Types (per backend)                         │
│  • ContainerResource (enroot on LocalRunner; container on Slurm/Cloud)   │
│  • VLLMResource (vLLM server process / job / service)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

**Layers:**

1. **Consumers**: `BaseTool` / `BaseReward` — request resources by kind and backend, then call them (MCP or input-text).
2. **ResourceEngine**: Single facade; holds **one heterogeneous store** of free resources (keyed by spec + backend) and arranges acquire/release/start/end directly—**no separate pool layer**. Implements start/acquire/release/monitor/control and optional dynamic scaling.
3. **Runners (backends)**: `LocalRunner`, `SlurmRunner`, `AWSRunner`, `K8sRunner` — each knows how to start, monitor, control, and end a resource of a given type on that infrastructure.
4. **Resource types**: Concrete resources (e.g. container handle, vLLM endpoint) created and managed by a runner.

**Single store, no pool layer:** The engine does not use a separate `ResourcePool` class. It keeps one store: a dict mapping `(spec, backend)` to a list of free resources. Any resource type (container, vLLM, etc.) lives in this store. The engine creates resources via runners and arranges acquire/release directly.

---

## 3. Resource Types

| Type              | Description                    | Call interface        | Notes                    |
|-------------------|--------------------------------|------------------------|--------------------------|
| **Container**     | Enroot (or Docker) container   | input-text (e.g. HTTP, exec) | CPU, memory, GPUs configurable |
| **VLLM**          | vLLM deployed model           | MCP tool-call or HTTP | Endpoint / MCP server    |

- **Container**: Execute commands inside container (e.g. code run); control CPU cores, memory, GPUs.
- **VLLM**: Inference endpoint; can be called via MCP server (tool-call) or direct text/HTTP.

---

## 4. Backends (Runners)

| Backend        | Scope        | Container support     | VLLM support              |
|----------------|-------------|------------------------|----------------------------|
| **LocalRunner**| Single node | Enroot (current path) | Local process / subprocess |
| **SlurmRunner**| Cluster     | Enroot in job         | vLLM in Slurm job          |
| **AWSRunner**  | Cloud       | ECS / EKS container   | SageMaker / EC2 / EKS      |
| **K8sRunner**  | Kubernetes  | Pods                  | Deployment / Service       |

Each runner implements the same **Runner** interface (see class templates).

---

## 5. ResourceEngine Interface (High-Level)

- **Lifecycle**
  - `start(resource_spec, size=1, backend=...)` — start a pool for a resource kind (e.g. container image + limits) on the given backend.
  - `acquire(id, resource_spec, backend=...)` — acquire a resource instance for `id` (same id → same instance).
  - `release(resource, id)` — return resource to pool.
  - `reset(resource, args=...)` — reset resource state (e.g. container reset).
  - `aclose()` — shut down all pools and backends.

- **Scaling**
  - **Dynamic scaling**: Optional component that, based on request frequency or current utilization, calls `start(..., size=N)` to scale up or scales down by ending excess resources.

- **Monitoring**
  - `monitor(resource)` or `get_status(resource)` — execution state: running / success / fail; optional metrics (CPU, memory, last error).

- **Control**
  - `control(resource, **kwargs)` — set or update container limits (memory, cpu_cores, gpus) where supported by the backend.

- **Calling a resource**
  - **MCP tool-call**: Resource exposes an MCP server; engine (or consumer) invokes tools via MCP (e.g. for vLLM tool-use).
  - **Input-text**: Generic input (e.g. HTTP body, stdin) — used for code execution in container, or simple HTTP to vLLM.

Tools/rewards use the engine to **acquire** a resource, then call it via the appropriate **call adapter** (MCP or input-text), then **release**.

---

## 6. Integration with Existing Code

- **EnvironmentManager** can become a **thin wrapper** over ResourceEngine:  
  `EnvironmentManager` → `ResourceEngine` with backend=`LocalRunner`, resource_type=container (enroot), and existing `BaseEnv` implementations (e.g. `PythonSandboxEnv`) backed by a `ContainerResource` from the engine.
- **BaseTool / BaseReward**: Replace direct `EnvironmentManager.start/acquire/release` with `ResourceEngine.start/acquire/release`; pass a **resource spec** (e.g. image, limits) and optional backend. Calling the resource uses the chosen **call interface** (MCP or input-text) as defined in the proposal.

---

## 7. Container Backend: Enroot

- For **LocalRunner**, containers are implemented with **enroot** (current approach in `envs/manager/enroot.py`).
- Resource limits (CPU, memory, GPUs) are applied via enroot (and systemd-run where available), as in the existing `_systemd_prefix` and `Containers.run` logic.
- The engine’s **ContainerResource** wraps the same enroot `Container` / client so that control (e.g. update limits) and exec (input-text) remain consistent.

---

## 8. Summary

| Concept           | Role                                                                 |
|------------------|----------------------------------------------------------------------|
| **ResourceEngine** | Single facade; pools and backends; start/acquire/release/monitor/control; optional dynamic scaling. |
| **Runner**         | Backend: how to start, monitor, control, end a resource (Local/Slurm/AWS/K8s). |
| **Resource type**  | Container (enroot), VLLM; each runner supports one or both.          |
| **Call interface** | MCP tool-call (MCP server), input-text (HTTP, exec, etc.).           |
| **Tools/Rewards**  | Use ResourceEngine to get a resource, then call via MCP or input-text. |

This keeps the current tool/reward flow (acquire → use → release) while generalizing the underlying manager to multiple resource types and backends, with a clear path to dynamic scaling and multiple ways to “call” a resource.
