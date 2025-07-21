# Installation

To install, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/agentfly/agentfly.git
cd agentfly
```

2. Initialize and install dependencies

```bash
git submodule init
git submodule update

pip install -r agents/requirements.txt
pip install -r verl/requirements.txt
```
3. Optional
```bash
conda install conda-forge::redis-server==7.4.0
```


