# ALFWorld Episode Reward

::: agentfly.rewards.alfworld_reward.alfworld_episode_reward
    options:
      show_source: true

## Description

Evaluates agent performance in ALFWorld tasks by checking the episode completion status and reward from the environment state. Returns a dict with `reward` (the environment's current reward value).

**Decorator Configuration:**
- **name**: `"alfworld_episode_reward"`
- **resource_spec**: `ALFWorldSpec`
- **backend**: `"local"`

## Technical Details

**Implementation:**
- Steps environment with empty action to get current state
- Extracts reward value from environment response
- Handles None reward values by defaulting to 0.0
- Provides debug output for reward values

**Use Cases:**
- Evaluating task completion in household environments
- Training agents on multi-step instruction following
- Measuring progress in text-based interactive environments

**Example Usage:**

```python
from agentfly.core import Context
from agentfly.rewards.alfworld_reward import alfworld_episode_reward

# Inside a rollout, `context` is injected and passed through to rewards.
result = await alfworld_episode_reward(
    prediction="take apple",
    context=context,
)
print(result)  # {"reward": 0.0} or {"reward": 1.0} if task completed
```

**Environment Integration:**
- Requires active ALFWorld environment instance
- Uses environment's internal reward mechanism
- Suitable for episodic task evaluation
