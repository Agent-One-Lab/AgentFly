# ALFWorld Reward

::: agentfly.rewards.alfworld_rewards.alfworld_reward
    options:
      show_source: true

## Description

The outcome reward for ALFWorld episodes, registered as `"alfworld_reward"`. It reads the
episode's stored `won` flag from the ALFWorld environment the tools used, without taking another
step, and returns:

- `reward`: `10.0` if the task was completed, else `0.0` (`10 * won`, verl-agent's scale).
- `trajectory/accuracy`: `1.0` or `0.0`, logged as the per-trajectory success rate.

The 0.1 invalid-action penalty is applied separately by the GiGPO estimator, not by this reward.

**Example Usage:**

```python
from agentfly.rewards import alfworld_reward

# The agent calls the reward after each episode with the episode's context.
result = await alfworld_reward(context=context)
print(result)  # {"reward": 0.0, "trajectory/accuracy": 0.0}, or 10.0 / 1.0 if the task was completed
```

See [ALFWorld Rewards](../resources/alfworld/rewards.md) for how it fits with the `alfworld_step`
tool and a training script.
