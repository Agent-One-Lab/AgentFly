# Migration: EnvironmentManager → Resource Engine

Suggestions for migrating from `EnvironmentManager` (env + warm pool) to the `ResourceEngine` (single heterogeneous store, no pool layer). **No code changes are made in this document**—only a migration plan and options.

---

## 1. Current vs Target

| Aspect | Current (EnvironmentManager) | Target (ResourceEngine) |
|--------|------------------------------|--------------------------|
| **Key** | `env_cls` (type) | `(ResourceSpec, backend)` |
| **Unit** | `BaseEnv` (start, reset, **step**, aclose) | `BaseResource` (start, reset, get_status, control, end, aclose) |
| **Consumers** | Tools/rewards use `env_cls`, `pool_size`, `env_kwargs`; call `env.step(action)` | Could keep env API via adapter, or switch to `resource_spec` + `CallInterface` |
| **Storage** | One `WarmPool` per env class | One engine store: `_free[pool_key]` = list of any resource type |

Main gap: **BaseEnv has `step(action)`**; tools/rewards call `await env.step(action)`. **BaseResource** has no `step`—invocation is via **CallInterface** (MCP or input-text). So migration must either preserve a `step`-like API (adapter) or move consumers to the resource engine’s call interface.

---

## 2. Migration Strategies

### Option A: Adapter — Keep BaseEnv API, Engine Under the Hood (recommended for minimal churn)

**Idea:** `EnvironmentManager` becomes a thin wrapper over `ResourceEngine`. Consumers still use `env_cls`, `pool_size`, `env_kwargs` and still receive a `BaseEnv`; under the hood the engine manages resources and an adapter turns a `BaseResource` into a `BaseEnv`.

**Steps:**

1. **Map env class + kwargs → ResourceSpec + backend**
   - For each env class that is “container-based” (e.g. `PythonSandboxEnv`, `ScienceWorldEnv`, `WebAgentTextEnv`, `ALFWorldEnv`), define a function or registry that builds a `ResourceSpec` from `env_kwargs` (e.g. `image`, `cpu`, `mem` → `ResourceSpec(kind=CONTAINER, image=..., cpu_count=..., mem_limit=...)`).
   - Choose a default `backend` (e.g. `"local"`) for existing envs. Optionally make backend configurable per env class later.

2. **Introduce an EnvAdapter(BaseEnv)**
   - Holds a `BaseResource` (e.g. `ContainerResource`) and the `ResourceSpec` / backend used to acquire it.
   - Implements `BaseEnv`:
     - `start` / `reset` / `aclose` delegate to the underlying resource’s `start` / `reset` / `end`.
     - **`step(action)`**: implement by “calling” the resource via the appropriate mechanism:
       - For container-based envs that today use HTTP (e.g. PythonSandboxEnv): adapter needs the container’s endpoint (e.g. from resource or spec) and performs an HTTP request with `action` as input (same as current env’s step).
       - For container-based envs that use `exec_run` (e.g. raw command): use `resource.container.exec_run(action)` (or equivalent) and return output as observation.
   - So “step” = one of: HTTP to container, exec in container, or later MCP call. The adapter hides whether the backend is engine-managed container or legacy env.

3. **EnvironmentManager implementation**
   - `start(env_cls, size, env_kwargs)`:
     - Compute `ResourceSpec` + `backend` from `env_cls` + `env_kwargs` (using the mapping above).
     - Call `ResourceEngine.start(spec, size=size, backend=backend)` (using the global engine or an injected one).
   - `acquire(env_cls, id)`:
     - Compute `spec` + `backend` from `env_cls` and the **same** `env_kwargs` that were used for `start` (must be stored per env_cls or passed in).
     - Call `engine.acquire(id, spec, backend)` → get `BaseResource`.
     - Wrap it in `EnvAdapter(resource, spec, backend)` (or a factory that knows how to implement `step` for this env_cls) and return as `BaseEnv`.
   - `release(env, id)`:
     - If `env` is an `EnvAdapter`, unwrap to get the underlying resource.
     - Call `engine.release(resource, id)`.
   - `reset(env, env_args)`:
     - Unwrap if adapter, then `engine.reset(resource, *args, **env_args)`.
   - `aclose()`:
     - Call `engine.close()` (or iterate and release/end all resources that were created via this manager).

4. **Where to store env_kwargs for acquire**
   - Today `start(env_cls, size, env_kwargs)` is called once; `acquire(env_cls, id)` has no kwargs. So the engine must know the “current” spec for that env_cls. Options:
   - **A1:** EnvironmentManager keeps a side map `_spec_for_env_cls: type[BaseEnv] -> (ResourceSpec, backend)` (and optionally env_kwargs) set in `start`, and used in `acquire`.
   - **A2:** Tools/rewards pass `env_kwargs` again on acquire (e.g. `acquire(env_cls, id, env_kwargs)`). Then no per-env_cls state in the manager, but all call sites must pass kwargs.

5. **Per-env-class step semantics**
   - Each current env class has its own `step` (e.g. HTTP to /run, or exec, or game action). The adapter (or a small registry keyed by env_cls) should dispatch `step(action)` to the right behavior (HTTP vs exec vs future MCP). So either:
   - One **EnvAdapter** class that takes an optional “step strategy” (e.g. callable or enum: “http”, “exec”, “mcp”), or
   - Subclasses like `PythonSandboxEnvAdapter`, `ScienceWorldEnvAdapter`, etc., each implementing `step` for that env’s protocol.

**Pros:** Tools and rewards stay unchanged (same `env_cls`, `pool_size`, `env_kwargs`, same `env.step(action)`). Only EnvironmentManager and a new adapter layer change.  
**Cons:** Adapter and env_cls→spec mapping must be maintained; long term you may want to move to Option B.

---

### Option B: Direct migration — Tools/Rewards use ResourceEngine + CallInterface

**Idea:** Tools and rewards stop using `env_cls` and `EnvironmentManager`. They declare `resource_spec` (and optionally `backend`); they acquire a `BaseResource` from the engine and call it via **CallInterface** (input-text or MCP) instead of `env.step(action)`.

**Steps:**

1. **BaseTool / BaseReward API change**
   - Add (or replace) attributes: e.g. `resource_spec: ResourceSpec`, `backend: str = "local"`, `pool_size` (or rely on engine’s `start(spec, size, backend)`).
   - Remove or deprecate `env_cls` and `env_kwargs` for stateful tools/rewards that today use an env.

2. **Environment lifecycle in tool/reward**
   - Replace `EnvironmentManager.start(env_cls, size, env_kwargs)` with `get_engine().start(spec, size=pool_size, backend=backend)`.
   - Replace `EnvironmentManager.acquire(env_cls, id)` with `get_engine().acquire(id, spec, backend)` → get `BaseResource`.
   - Replace `env.step(action)` with a call through the appropriate **CallInterface**:
     - For “input-text” (e.g. code execution, game action): e.g. `input_text_adapter.call(resource, action)` which for a container might do HTTP or `resource.container.exec_run(...)` and return the result as the “observation”.
   - Replace `EnvironmentManager.release(env, id)` with `get_engine().release(resource, id)`.
   - Replace `EnvironmentManager.reset(env, env_args)` with `engine.reset(resource, *args, **env_args)`.

3. **Implement InputTextCall (and optionally MCP) for containers**
   - For each current env “flavor” (Python sandbox HTTP, ScienceWorld, Webshop, ALFWorld, etc.), define how an action string is turned into a call (HTTP URL/body, or exec command). The engine or a small helper can hold the right call implementation per resource type/spec so tools/rewards just pass `action` and get back the observation.

4. **Deprecate or remove EnvironmentManager and BaseEnv**
   - Once all stateful tools/rewards use the engine + call interface, `EnvironmentManager` and the old env pool can be removed or kept only for backward compatibility.

**Pros:** Single model (resource engine + call interface); no adapter; aligns with the “decoupled resource engine” design.  
**Cons:** Touches every stateful tool and reward; more invasive.

---

### Option C: Hybrid — Adapter first, then gradual direct migration

1. Implement **Option A** so that existing code runs on the resource engine without behavior change.
2. Optionally add a **parallel path** in BaseTool/BaseReward: e.g. if `resource_spec` is set, use the engine directly and CallInterface; if only `env_cls` is set, use EnvironmentManager (which internally uses the engine + adapter).
3. Migrate tools/rewards one by one from `env_cls` to `resource_spec` + call interface; when none use `env_cls`, remove EnvironmentManager and the adapter.

---

## 3. Mapping env_cls + env_kwargs → ResourceSpec (for Option A / C)

- **PythonSandboxEnv**(image=..., cpu=..., mem=..., ...)  
  → `ResourceSpec(kind=CONTAINER, image=..., cpu_count=..., mem_limit=..., ports={...}, environment=...)`  
  Step adapter: HTTP to container (e.g. `/run` or similar) with body = code/action.

- **ScienceWorldEnv**, **WebAgentTextEnv**, **ALFWorldEnv**  
  → Same idea: `ResourceSpec(CONTAINER, image=..., ...)`. Step adapter: HTTP or exec according to how the current env’s `step` works (server endpoint vs shell command).

- **RedisEnv**, **ChessPuzzleEnv**  
  → If they don’t use a container, they could stay on the old pool for now, or be modeled as a “local” resource with a trivial spec (e.g. process or in-memory) and a small adapter that delegates `step` to the existing env implementation.

Keep a single place (module or registry) that implements `env_cls + env_kwargs → (ResourceSpec, backend)` so EnvironmentManager and tests stay consistent.

---

## 4. Concrete Checklist (Option A)

1. Add **env_cls → (ResourceSpec, backend)** mapping (and optionally **env_cls → step strategy** for the adapter).
2. Implement **EnvAdapter(BaseEnv)** (or one per env type) that wraps a `BaseResource` and implements `step(action)` via HTTP/exec/MCP.
3. Change **EnvironmentManager** to use `get_engine()` and the mapping; implement `start` / `acquire` / `release` / `reset` / `aclose` as in section 2 Option A. Ensure **env_kwargs** are stored (e.g. in `_spec_for_env_cls`) so `acquire` can resolve the same spec.
4. Ensure **ResourceEngine** is initialized with the right runners (e.g. LocalRunner with enroot) before any tool/reward runs.
5. **Tests:** Run existing env/tool/reward tests; add a test that EnvironmentManager returns an env that passes BaseEnv contract and that release returns the resource to the engine (e.g. re-acquire and get same or another resource).

---

## 5. Summary

- **Minimal change, same API:** Use **Option A** (adapter + EnvironmentManager over ResourceEngine). One store, one engine; tools/rewards keep using `env_cls` and `env.step(action)`.
- **Full alignment with resource engine:** Use **Option B** (tools/rewards use `resource_spec` + engine + CallInterface). No EnvironmentManager, no BaseEnv in the loop.
- **Pragmatic:** Use **Option C** (adapter first, then migrate consumers gradually to Option B and eventually remove EnvironmentManager and the adapter).

No code changes are included in this document; the above is a plan only.
