# Installation

**Install With Script**

To install dependencies, run the following script in conda environment. We default to use python3.10.
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

2. Initialize and install dependencies

Basic python packages installation:
```bash
pip install -e .
pip install -e '.[verl]' --no-build-isolation
```

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
Search requires redis to cache results, an optional way to install with conda:
```bash
conda install conda-forge::redis-server==7.4.0
```

