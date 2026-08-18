# ALFWorld Tools

ALFWorld is driven by a single tool, `alfworld_step`. The agent takes one action
per turn; each result carries the new observation **with the admissible actions
for that state appended**, so the model can always pick a valid action without a
separate lookup call.

## Tools Reference

### alfworld_step

::: agentfly.tools.src.alfworld.tools.alfworld_step
    options:
      show_source: true

**Description:** Take an action in the ALFWorld environment. Returns a dict with
`observation`, `reward`, `done`, and `info`. The `observation` string is the raw
environment feedback followed by an `Admissible actions: [...]` list for the
current state. Example actions: `"go to fridge 1"`, `"take apple 1 from countertop 1"`,
`"open fridge 1"`, `"look"`.

The environment resets automatically on first use to the episode's task
(`context.metadata["task_id"]`); there is no separate reset tool.

## Usage with an agent

```python
from agentfly.tools import alfworld_step
from agentfly.rewards import alfworld_episode_reward

tools = [alfworld_step]

system_prompt = (
    "Navigate the ALFWorld environment and complete tasks by interacting with "
    "objects. Use alfworld_step to act. Each observation lists the admissible "
    "actions for the current state under 'Admissible actions:' — always choose "
    "your action from that list."
)
```

The reward (`alfworld_episode_reward`) reads the episode's terminal `won` flag
from the same acquired resource and returns `10 * won`.

## Tool Configuration

`alfworld_step` acquires an ALFWorld container from the `ResourceEngine` via
`context.acquire_resource(spec=ALFWorldSpec, scope="global", backend="local")`.
The maximum concurrent environments is controlled by `ALFWorldSpec.max_global_num`.
Pooling/sharing is handled by `Context`/`ResourceEngine`; you don't pass explicit
`env` instances to the tool.
