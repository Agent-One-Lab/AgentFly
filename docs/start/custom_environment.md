# Build a Custom Environment

A plain tool is a function: every call starts from nothing. When calls need to share state, such
as a game board, a sandbox or a browser session, put that state in an **environment** and let the
tools use it through the rollout's `Context`. AgentFly creates environment instances, gives each
rollout its own, pools them for reuse, and ends them when you're done.

This page builds a number-guessing environment: the agent guesses a secret number and is told
"higher" or "lower". The full example is in
[`examples/custom_env/`](https://github.com/Agent-One-Lab/AgentFly/tree/main/examples/custom_env):

```
examples/custom_env/
├── my_task/
│   ├── env.py       # the environment and its spec
│   └── tools.py     # the tool and the reward that use it
├── make_data.py     # writes data/guess_{train,val}.json
├── run.py           # runs an agent on two tasks
└── train.sh         # trains on the environment
```

## 1. The environment

Subclass `BaseEnv` and implement its lifecycle methods. The engine calls `start` once when it
creates an instance and `aclose` when it destroys it; tools call `reset` and `step`.

```python
--8<-- "examples/custom_env/my_task/env.py:env"
```

The **spec** tells the engine how to create instances. `LocalEnvResourceSpec` runs the env inside
the current Python process, the lightest option. For an environment that runs in a container, use a
`ContainerResourceSpec` with an image instead, as the built-in Python sandbox, ALFWorld and WebShop
environments do (see [Resources](../features/resources.md#resource-types)).

`env_cls_path` must be importable by that dotted path wherever the tools run, including the
training workers (see [Training](#4-training)).

## 2. Tools and rewards that use it

A tool gets the environment by declaring a `context` parameter. AgentFly passes the rollout's
`Context` there and hides the parameter from the model; dataset fields of the task are in
`context.metadata`.

```python
--8<-- "examples/custom_env/my_task/tools.py:get_env"
```

```python
--8<-- "examples/custom_env/my_task/tools.py:tool_reward"
```

Within a rollout, every `acquire_resource` call with the same spec returns the **same instance**,
so the tool's guesses and the reward's check see the same game. Different rollouts get different
instances, even when they run concurrently.

### `scope`: reuse or recreate

`scope` decides what happens to the instance when the rollout ends:

| `scope` | At the end of the rollout | Use when |
|---|---|---|
| `"global"` | The instance goes back to the pool, still running, and a later rollout reuses it. It is **not** reset for you. | Starting is slow (containers, loaded data) and `reset` can fully clear the state. Reset it on first use in each rollout, as `get_env` does with `context.is_spec_acquired`. |
| `"rollout"` (default) | The instance is ended (`aclose`), and the next rollout creates a new one. | Starting is cheap, or state can't be cleared reliably. |

Pooled instances are ended when the process calls `await ResourceEngine.close()` (training does
this when it finishes; `run.py` does it at the end).

The tool and the reward must use the same spec and the same scope to share an instance. A rollout
can also use several **different** environments (or an environment plus a container): each kind gets
its own instance. If you pass `id=` yourself, use a different id for each kind; reusing one id for
two kinds raises an error.

## 3. Run an agent on it

```python
--8<-- "examples/custom_env/run.py:run"
```

```bash
cd examples/custom_env && python run.py
```

Output from a run (each target is played twice, and every attempt has its own game):

```
target=37 reward=1.0 guesses=7
target=37 reward=1.0 guesses=3
target=81 reward=1.0 guesses=6
target=81 reward=0.0 guesses=1
```

## 4. Training

Training loads the tool, the reward and the env by import path, in the Ray workers as well, so
`my_task` must be importable there. The example script puts it on `PYTHONPATH` and refers to the
tool and reward as `my_task.tools:guess` and `my_task.tools:guess_reward`:

```bash
python examples/custom_env/make_data.py      # writes examples/custom_env/data/guess_{train,val}.json
bash examples/custom_env/train.sh            # extra Hydra overrides can be appended
```

```bash
--8<-- "examples/custom_env/train.sh"
```

Each training rollout gets its own environment instance from the pool. `max_global_num` in the
spec caps how many exist at once; rollouts beyond that wait for a free instance, so set it to at
least the number of rollouts you want to run concurrently (`train_batch_size × num_chains` for full
parallelism).
