# Configuration

AgentFly is configured through a small set of environment variables, all under a
single **`AF_`** prefix. Paths derive from one root (`AF_HOME`); the rest are
behavior toggles. Nothing here is required — every variable has a sensible
default.

## Directories

All per-user state lives under `AF_HOME` (default `~/.agentfly`), following the
`~/.<tool>` convention. The cache, data, and config directories derive from it.

| Variable | Default | Purpose |
|---|---|---|
| `AF_HOME` | `~/.agentfly` | Root for all per-user state. Set this to relocate everything at once. |
| `AF_CACHE_DIR` | `$AF_HOME/cache` | Downloaded artifacts (search corpus + FAISS index, enroot images, jinja templates). |
| `AF_DATA_DIR` | `$AF_HOME/data` | Runtime data (e.g. Redis data dir). |
| `AF_CONFIG_DIR` | `$AF_HOME/config` | User-editable config files. Packaged defaults (`search.yaml`, `redis/redis.conf`, `vllm.yaml`, `code.yaml`) are seeded here on first run if missing — edit `~/.agentfly/config/search.yaml` to set your search API keys. |

!!! note "Config lives under `AF_HOME` now"
    In earlier versions `AF_CONFIG_DIR` pointed *inside the installed package*.
    It now defaults to `~/.agentfly/config`, seeded from the packaged defaults on
    first import (existing files are never overwritten). Put your API keys in
    `~/.agentfly/config/search.yaml`, not in `site-packages`.

## Behavior toggles

| Variable | Default | Purpose |
|---|---|---|
| `AF_ASSUME_YES` | unset | Accept data-download approval prompts non-interactively (CI/batch). Set to `1`. |
| `AF_CONTAINER_ENGINE` | `enroot` | Container backend when a spec doesn't pin one: `enroot`, `docker`, or `daytona`. |
| `AF_NO_STREAM` | unset | Disable streaming in LLM backends (set `1` to force non-streaming requests). |
| `AF_CONN_RETRIES` | `4` | Connection retry attempts for LLM backend requests. |
| `AF_RAY_GET_DEFAULT_TIMEOUT_SEC` | `7200` | Default timeout for blocking `ray.get` calls in Ray-backed container resources. |
| `AF_SKILLS_ROOT` | repo default | Root directory the skills loader resolves skill manifests from. |

## Tool / environment credentials

These follow each tool's own convention and are read directly:

- **Search** — `search.yaml` (`SERPER_API_KEY`, `GOOGLE_API_KEY`, `CUSTOM_SEARCH_ENGINE_ID`) under `AF_CONFIG_DIR`; `RETRIEVER_*` env vars override the retriever server's corpus/index paths and host/port (see [CLI](cli.md#agentfly-search)).
- **Daytona** — `DAYTONA_API_KEY` (required for the `daytona` engine), `DAYTONA_API_URL`, `DAYTONA_TARGET` (the Daytona SDK's own variables).

## Daytona sandbox limits

The Daytona engine's per-sandbox ceilings and lifecycle timeouts are **not**
environment variables — they are a config object, `DaytonaResourceSpec`, carried
on the container spec:

```python
from agentfly.resources.types import ContainerResourceSpec, DaytonaResourceSpec

spec = ContainerResourceSpec(
    category="container",
    container_engine="daytona",
    dockerfile="FROM ubuntu:22.04\n",
    daytona=DaytonaResourceSpec(
        max_cpus=4, max_memory_gb=8, max_disk_gb=10,
        auto_stop_min=30, auto_delete_min=60,
        snapshots="auto",  # "auto" | "on" | "off"
    ),
)
```

Omit `daytona=` to take the defaults shown above.
