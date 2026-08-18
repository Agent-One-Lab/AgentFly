# Installation

## Requirements

| | |
|---|---|
| **OS** | Linux only — enroot is Linux-only; Debian/Ubuntu and RHEL/CentOS/Fedora are both supported (install commands below). |
| **Python** | 3.12.x — `pyproject.toml` enforces `>=3.12,<3.13`. Newer or older versions will fail to install. |
| **GPU** | NVIDIA GPU(s) with CUDA. Required by vLLM and the verl trainer. The shipped `train_*.sh` scripts target 1 to 8 GPUs (`trainer.n_gpus_per_node=1` for `train_example.sh`, up to `=8` for the bigger tasks); per-GPU VRAM scales with model size (Qwen 3B fits comfortably on a single 80 GB card; 32B benefits from multi-GPU). |
| **Sudo** | Needed once, to install **enroot** (used by `code_interpreter`, `alfworld`, `webshop`, `scienceworld`, and any container-backed tool). Skip if you only need the lighter tools (`calculator`, retrieval API, etc.). |
| **Disk** | A few hundred GB headroom recommended: HuggingFace model weights, enroot images under `$AF_CACHE_DIR/enroot/`, and (for retrieval tasks) the Wikipedia corpus + FAISS index downloaded into `$AF_CACHE_DIR/data/search/` (~30 GB). Defaults live under `~/.agentfly/cache/`. |
| **uv** | The reference setup uses [uv](https://docs.astral.sh/uv/) (`bash install.sh` installs it if missing). uv provisions Python 3.12, manages a project-local `.venv`, and installs from the committed `uv.lock`. |

If you only want to **run** agents — no RL training, no container-backed tools — `uv sync` alone is enough. The `[verl]` extra and the enroot install are training- and environment-specific.

**Install With Script**

To install everything, run the script below. It installs [uv](https://docs.astral.sh/uv/) if it is not already present, provisions Python 3.12, and creates a project-local `.venv`.
```
bash install.sh
```


**Step-by-Step Installation**

Alternatively, you can customize the installation by following these steps:

1. Clone the repository and initialize submodules:

    ```bash
    git clone https://github.com/Agent-One-Lab/AgentFly
    cd AgentFly
    git submodule init
    git submodule update
    ```

2. Install dependencies with uv

    uv creates a project-local `.venv`, pins Python 3.12, and installs from the committed `uv.lock`. Install the base package, or add the `verl` extra for RL training:

    ```bash
    uv sync                 # base: run agents
    uv sync --extra verl    # base + RL training (verl)
    ```

    Then activate the environment (`source .venv/bin/activate`) or prefix commands with `uv run`.

    Some of our tools & environments are managed by *enroot* backend. To use them, please install [enroot](https://github.com/NVIDIA/enroot/blob/master/doc/installation.md) (sudo required). Such tools include code_interpreter, retrieval, webshop, alfworld, sciencworld.

    ```bash
    # enroot install
    # Debian-based distributions
    arch=$(dpkg --print-architecture)
    curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot_3.5.0-1_${arch}.deb
    curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot+caps_3.5.0-1_${arch}.deb # optional
    sudo apt install -y ./*.deb

    # RHEL-based distributions
    arch=$(uname -m)
    sudo dnf install -y epel-release # required on some distributions
    sudo dnf install -y https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot-3.5.0-1.el8.${arch}.rpm
    sudo dnf install -y https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot+caps-3.5.0-1.el8.${arch}.rpm # optional
    ```

3. Optional

    Search-backed tools use redis to cache results. Install a `redis-server` binary with your system package manager if you need them:

    ```bash
    # Debian/Ubuntu
    sudo apt-get install -y redis-server
    ```
