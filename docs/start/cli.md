# CLI Reference

The `agentfly` command-line interface is a small dispatcher (`src/agentfly/cli.py`) that hands argv off to one of four subcommands. Invoke it as either:

```bash
python -m agentfly.cli <command> [args...]
# or, if installed as a script:
agentfly <command> [args...]
```

`agentfly --help` lists the available commands. The four subcommands are:

| Command | Purpose | Style |
|---|---|---|
| [`train`](#agentfly-train) | RL training (verl PPO) | Hydra |
| [`deploy`](#agentfly-deploy) | Print/run a `vllm serve` command for local inference | click flags |
| [`swebench`](#agentfly-swebench) | SWE-Bench-style evaluation over a JSON dataset | click flags |
| [`search`](#agentfly-search) | Standalone dense-retriever HTTP server | env vars |

---

## `agentfly train`

Runs `agentfly.verl.trainer.main_ppo` with Hydra-style config overrides. All arguments after `train` are forwarded to Hydra; there are no flags parsed by the CLI itself.

```bash
agentfly train \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/rlhf/math/gsm8k_train.json \
    agent.init_config.agent_type=hf \
    agent.init_config.tools="[calculator]" \
    ...
```

The full set of override keys you'll typically use is documented in [Hydra Config](hydra_config.md). For an end-to-end walkthrough, see [First Training](first_training.md). The upstream Hydra schema lives in `verl/verl/trainer/config/ppo_trainer.yaml`.

---

## `agentfly deploy`

Prints and executes a `vllm serve` command for serving the model with an OpenAI-compatible API plus chat-template injection (so tool-calling models like Qwen2.5 work out of the box). Useful for setting up a local LLM endpoint that agents can hit via the `client` backend.

```bash
agentfly deploy \
    --model-name-or-path Qwen/Qwen2.5-VL-3B-Instruct \
    --template qwen2.5-vl-system-tool \
    --tp 2 --dp 2 \
    --port 8000
```

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--model-name-or-path` | str | — | HuggingFace model id or local path. |
| `--template` | str | `None` | `chat-bricks` template name. If set, the template's Jinja is written under `$AGENT_DATA_DIR/cache/jinja_template.jinja` and passed to `vllm serve` via `--chat-template`. |
| `--tp` | int | `1` | Tensor-parallel size. |
| `--pp` | int | `1` | Pipeline-parallel size. |
| `--dp` | int | `1` | Data-parallel size. |
| `--gpu-memory-utilization` | float | `0.8` | Forwarded to `vllm serve --gpu-memory-utilization`. |
| `--tool-call-parser` | str | `hermes` | Forwarded to `vllm serve --tool-call-parser`. |
| `--port` | int | `8000` | Forwarded to `vllm serve --port`. |
| `--allowed-local-media-path` | str | `None` | Forwarded to `vllm serve --allowed-local-media-path` (for vision models that load local images). |

The command emits and runs:

```
vllm serve <model> [--chat-template ...] --trust-remote-code \
    --tensor-parallel-size <tp> --pipeline-parallel-size <pp> \
    --data-parallel-size <dp> --port <port> \
    --gpu-memory-utilization <util> \
    --enable-auto-tool-choice --tool-call-parser <parser> \
    [--allowed-local-media-path ...]
```

---

## `agentfly swebench`

Runs an SWE-Bench-style evaluation: loads a JSON dataset of issues, runs an agent (Bash- or Qwen3-Coder-style) over each, evaluates with a registered reward, and writes one JSON per sample plus a `run_summary.json`.

```bash
agentfly swebench \
    --data-path ./data/rlhf/os/swe-bench-verified.json \
    --result-dir ./results/swe/run-2026-04 \
    --model-name-or-path Qwen/Qwen3-32B-Coder \
    --agent qwen3_coder \
    --tool-set bash \
    --reward-name r2e_gym_reward \
    --backend client \
    --vllm-base-url http://localhost:8000/v1
```

### Required

| Flag | Purpose |
|---|---|
| `--data-path` | JSON file: a list of instances, or `{"data": [...]}` / `{"instances": [...]}`. |
| `--result-dir` | Output directory; gets one JSON per sample plus `run_summary.json`. |
| `--model-name-or-path` | HF model id or local path. |

### Generation / rollout

| Flag | Default | Purpose |
|---|---|---|
| `--template` | `None` | Optional `chat-bricks` template name. |
| `--temperature` | `0.0` | Sampling temperature. |
| `--max-turns` | `30` | Per-sample turn cap. |
| `--num-chains` | `1` | Parallel chains per sample. |
| `--max-concurrent-chains` | unlimited | Cap concurrent chains across the batch. |

### Agent / tools / reward

| Flag | Choices / default | Purpose |
|---|---|---|
| `--agent` | `bash` \| `qwen3_coder` (default) | Which SWE agent class to use. |
| `--tool-set` | `bash` (default) \| `file` | `bash`: bash only. `file`: file workspace tools (Qwen3-Coder). |
| `--reward-name` | `r2e_gym_reward` (default) | Registered reward name (e.g. `r2e_gym_reward`, `swe_reward`). |

### Backend

| Flag | Choices / default | Purpose |
|---|---|---|
| `--backend` | `client` (default) \| `async_vllm` | Use a remote OpenAI-compatible endpoint or a local vLLM engine. |
| `--vllm-base-url` | `http://localhost:8000/v1` | Endpoint URL when `--backend client`. |
| `--api-key` | `EMPTY` | API key for the client backend. |
| `--tensor-parallel-size` / `--tp` | `1` | TP size when `--backend async_vllm`. |
| `--data-parallel-size` / `--dp` | `1` | DP size when `--backend async_vllm`. |
| `--max-model-len` | `None` | Optional max sequence length for vLLM. |

### Resources / scope

| Flag | Default | Purpose |
|---|---|---|
| `--resource-backend` | `local` | `Context` resource backend (e.g. `local`, `ray`). |
| `--enroot-images-path` | `None` | If set, sets `ENROOT_IMAGES_PATH` for container-backed rewards. |
| `--enroot-async / --no-enroot-async` | `--enroot-async` | Sets `ENROOT_ASYNC=1` (default) or `0`. |

### Sampling

| Flag | Default | Purpose |
|---|---|---|
| `--limit` | all | Run at most this many samples (after shuffle). |
| `--seed` | `None` | Shuffle dataset with this seed. |

`run_summary.json` ends with overall accuracy printed to stdout.

---

## `agentfly search`

Starts a FastAPI HTTP server wrapping the dense retriever (E5-base-v2 + FAISS over Wikipedia-18). Used as a remote retriever process so the GPU stays in one process while clients call via HTTP. The matching tool is `agentfly.tools.src.search.async_dense_retrieve_api`.

There are no CLI flags. Configuration is entirely via environment variables:

| Env var | Default | Purpose |
|---|---|---|
| `RETRIEVER_CORPUS_FILE` | `$AGENT_CACHE_DIR/data/search/wiki-18.jsonl` | Corpus file path. |
| `RETRIEVER_INDEX_FILE` | `$AGENT_CACHE_DIR/data/search/e5_Flat.index` | FAISS index file path. |
| `RETRIEVER_HOST` | `0.0.0.0` | Bind host. |
| `RETRIEVER_PORT` | `8765` | Bind port. |

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check + retriever-loaded status. |
| `POST` | `/search` | Body: `{"query": str, "top_k": int}`. Returns `{"results": [{"contents": str}]}`. |

Typical invocation:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
python -m agentfly.cli search
```

The corpus and FAISS index are auto-downloaded into `$AGENT_CACHE_DIR/data/search/` by `agentfly.tools.utils.data.download_tool_data("asyncdense_retrieve")` if they're not already present.

---

## Adding a New Subcommand

The dispatcher is a few-dozen-line file at `src/agentfly/cli.py`. Adding a new subcommand is two changes there:

1. Add a branch to the `if command == ...` chain that imports your target module and rewrites `sys.argv`.
2. Update the `--help` print block at the top.

The pattern is module-import based, so your subcommand can use any argument-parsing style (Hydra, click, argparse, plain `sys.argv`) — the dispatcher just hands argv off and calls `module.main()`.
