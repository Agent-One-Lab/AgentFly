# Build Your Own Task

This page walks end-to-end through training an agent on a new task. By the end you'll have a `train_<task>.sh` that wires your dataset, agent, tools, and reward into the verl PPO trainer. Every step links out to the deeper page that covers it.

## Files You'll Touch

For most tasks, you only create two files (the dataset and the training script) and reuse everything else from `agentfly.*`:

```
data/rlhf/<task>/<task>_train.json             ← your dataset
data/rlhf/<task>/<task>_val.json               ← (optional) validation split
my_task/tools.py                                ← only if you need a new tool
my_task/rewards.py                              ← only if you need a new reward
train_<task>.sh                                 ← your training script
```

Your tools and rewards live in your own files; you don't edit AgentFly's source. Training (and `AutoAgent.from_config`) loads them by import path: `"my_task.tools:my_tool"` (an importable module) or `"/abs/path/my_task/tools.py:my_tool"` (a file). When you construct an agent directly in Python, pass the objects themselves: `HFAgent(tools=[my_tool], reward_fn=my_reward)`.

The dataset JSON and the training script are the *minimum*. Tools, rewards, and agents are reusable across tasks — you only write new ones when nothing existing fits.

## Step 1: Prepare Your Data

Convert your source data into a JSON list of dicts. The minimum is `{"question": "..."}`; any additional fields you add (e.g. `answer`, `task_id`, `fen`) are kept as the task's metadata. A reward parameter with the same name receives the value (a parameter named `answer` gets the dataset's `answer`); tools read it from `context.metadata`.

See **[Data Preparation](data_preparation.md)** for the full convention, the predefined fields, and example shapes for math, QA, ALFWorld, ScienceWorld, WebShop, Chess, SWE-Bench, and multi-modal data.

## Step 2: Pick (or Subclass) an Agent

Most new tasks reuse an existing agent class:

| Agent | Use when | `agent_type` value |
|---|---|---|
| `HFAgent` | Standard tool-calling LLM agent — most common. | `hf` |
| `ReactAgent` | ReAct-style "thought → action" prompts. | `react` |
| `CodeAgent` | Agents that emit code blocks (e.g. for `code_interpreter`). | `code` |
| `ActionAgent` | Action-tag-style agents (e.g. `<action>...</action>`). | `action` |
| `GUIAgent` | Vision-grounded GUI tasks. | `gui` |
| `ImageEditingAgent` | Multi-modal image-editing tasks. | (specialized init) |
| `BashSWEAgent`, `Qwen3CoderSWEAgent` | SWE-Bench-style coding. | (specialized init) |

See **[Build an Agent](first_agent.md)** for usage and **[Agents API Reference](../api_references/agents/index.md)** for the full surface.

If none fit, subclass `BaseAgent` and override `parse(responses, context=None, **kwargs)`, which turns raw model outputs into assistant messages with their `tool_calls`. See the [custom agent example](../api_references/agents/index.md#custom-agent).

## Step 3: Define Tools (or Reuse Existing Ones)

If an existing tool fits — `calculator`, `code_interpreter`, `async_dense_retrieve`, `webshop_browser_action`, the alfworld/scienceworld tools — just import it from `agentfly.tools`. No new code needed.

For a new tool, decorate a function with `@tool`:

```python
from agentfly.tools import tool

@tool(name="my_tool", description="One-line description the model will see.")
async def my_tool(arg1: str, arg2: int = 1) -> str:
    """
    Args:
        arg1: First argument.
        arg2: Second argument with default.

    Returns:
        Observation string the model sees back.
    """
    return f"Observation for {arg1}"
```

A few things to note:

- The **tool name** in `@tool(name=...)` is the registered name (used in `agent.init_config.tools=[...]` in your training script). It's distinct from the Python identifier; pick something readable for the model.
- Return either a `str` or a `dict` containing `observation` (and optionally `image`, etc.).
- Async vs sync — both work. Async is preferred for I/O-bound tools.
- For tools that share state across calls (a game, a sandbox, a browser), see **[Build a Custom Environment](custom_environment.md)**.

## Step 4: Define a Reward

The reward function is what RL optimizes against. Decorate it with `@reward`. The framework auto-injects:

- `final_response` (string): the agent's final answer: the output of an answer tool (one declared with `status="finish"`, like `answer_math`) if the episode ended with one, otherwise the model's last message. Ordinary tool results (calculator, search, code) are never the final response, so leave `max_turns` room for the model to answer after its last tool call.
- `trajectory`: the full message list.
- `context`: the rollout `Context`, used for stateful rewards that acquire resources.
- **Any parameter name that matches a key in your dataset row** — this is how `answer`, `task_id`, etc. flow in.

```python
from agentfly.rewards import reward
from typing import List, Dict

@reward(name="my_task_reward")
def my_task_reward(final_response: str, answer: str, trajectory: List[Dict]) -> float:
    return 1.0 if final_response.strip() == answer.strip() else 0.0
```

The return value can be a `float` (gets wrapped into `{"reward": <value>}`) or a `dict` containing `reward` plus any extra metrics. Extra keys are logged as `reward_extra/<key>/{mean,max,min}` in WandB.

See **[Reward System](../features/reward_system.md)** for full conventions, including stateful rewards that share a resource with their tools.

## Step 5: Test Locally Before Training

Before launching RL training (which costs real GPU hours), confirm the data → tool → reward chain produces sane outputs. A short test:

```python
import pytest
from agentfly.agents import HFAgent
from agentfly.tools import calculator
from agentfly.rewards import reward

@reward(name="my_task_reward")
def my_task_reward(final_response: str, answer: str, trajectory):
    return 1.0 if final_response.strip() == answer.strip() else 0.0


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_my_task_smoke():
    agent = HFAgent(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        tools=[calculator],
        reward_fn=my_task_reward,
        template="qwen2.5",
        backend_config={"backend": "async_vllm"},
    )
    sample = {
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "answer": "4",
    }
    result = await agent.run(messages=sample, max_turns=3, group_size=1)
    for traj in result.trajectories:
        print(agent.extract_final_response(traj.segments[-1].messages), traj.reward)
```

Things to verify here:

- The agent **actually calls your tool** — for a chain rollout, `result.trajectories[0].segments[-1].messages` should contain at least one message with `role: "tool"`.
- The reward gets called and returns a **non-degenerate value** — not stuck at 0.0 or 1.0 for every sample.
- All your dataset's task-specific fields (`answer`, `task_id`, …) reach the reward — easy to confirm by adding a `print` inside the reward.

If the reward never sees a tool call or always returns the same value, fix that *before* burning training compute.

## Step 6: Wire Up a Training Script

Copy [`examples/custom_env/train.sh`](https://github.com/Agent-One-Lab/AgentFly/blob/main/examples/custom_env/train.sh), which trains a task whose tool, reward and environment live in your own module (see [Build a Custom Environment](custom_environment.md)), and change the parts specific to your task. The example scripts are in the repository (`git clone https://github.com/Agent-One-Lab/AgentFly`), not in the pip package.

| Override | What to set it to |
|---|---|
| `data.train_files` | Path to your training JSON. |
| `data.val_files` | Path to your validation JSON. |
| `agent.init_config.agent_type` | `hf`, `react`, `code`, `action`, etc. |
| `agent.init_config.tools` | List literal of tool names or import paths, e.g. `"[calculator,my_task.tools:my_tool]"`. |
| `agent.init_config.reward_name` | A reward name or import path, e.g. `my_task.rewards:my_reward`. |
| `agent.init_config.template` | Optional: a chat template name. Leave it unset to use the model's own chat template. |
| `agent.init_config.model_name_or_path` | Base model. |
| `trainer.experiment_name` | Unique name for WandB logging. |

Run `agentfly train` from the folder that contains your module: references like
`my_task.tools:my_tool` are imported in the trainer's Ray workers, which start in that folder, so
no `PYTHONPATH` is needed. File paths (`/path/to/tools.py:my_tool`) work from any folder. On a
multi-node cluster, the folder must be at the same path on every node (e.g. a shared filesystem).

The rest of the script (advantage estimator, KL coefficient, batch sizes, FSDP/vLLM flags, learning rate, etc.) typically stays the same as the template you copy from. The full key reference is in **[Hydra Config](hydra_config.md)**; the end-to-end flow is in **[First Training](first_training.md)**.

## Worked Example: GSM8K with the Calculator

A canonical "build your own task" example would be: an agent that solves GSM8K-style arithmetic problems by calling the calculator. **This already exists** in the repo — every piece is already provided:

| Step | What's there | Path |
|---|---|---|
| Data | GSM8K, shape `{question, answer}` | `data/rlhf/math/gsm8k_train.json` |
| Agent | Reuse `HFAgent` | `agent.init_config.agent_type=hf` |
| Tool | Reuse `calculator` | `agent.init_config.tools=[calculator]` |
| Reward | Reuse `math_equal_reward_tool` | `agent.init_config.reward_name=math_equal_reward_tool` |
| Script | The shipped reference script | `examples/train_scripts/train_example.sh` |

Run with `bash examples/train_scripts/train_example.sh`. That's the entire pipeline. To turn this into *your* task, swap one or more pieces:

- **Different dataset** → write `data/rlhf/<your_task>/<your_task>_train.json` and update `data.train_files`.
- **Different reward** → write `@reward(name="...")` and update `agent.init_config.reward_name`.
- **Different tool** → write `@tool(name="...")` and update `agent.init_config.tools`.
- **Different agent class** → set `agent.init_config.agent_type` (or subclass `BaseAgent`).

Each piece is independent: you can change one without touching the others. That's the whole design.

## Where to Go From Here

- **[Predefined Training Examples](../examples/predefined_training_examples.md)** — tasks AgentFly has been trained on, with training reports. Their scripts are under `examples/train_scripts/`; copy the closest one. **[Predefined Rewards](../api_references/rewards/predefined_rewards.md)** lists every built-in reward.
- **[Image Agent Example](../examples/image_agent.md)** — multi-modal walkthrough.
- **[Tool System](../features/tool_system.md)**, **[Reward System](../features/reward_system.md)**, **[Resources](../features/resources.md)** — for stateful tools and rewards, container management, and the full Tool/Reward contract.
