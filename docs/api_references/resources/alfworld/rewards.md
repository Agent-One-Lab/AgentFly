# ALFWorld Rewards

ALFWorld has one outcome reward, `alfworld_reward`. It is sparse: an episode earns **10** if the
agent completed the task and **0** otherwise.

## Reward Functions Reference

### alfworld_reward

::: agentfly.rewards.alfworld_rewards.alfworld_reward
    options:
      show_source: true

It reads the episode's stored `won` flag from the ALFWorld environment the tools used (it does not
take another step) and returns:

| Key | Value | Meaning |
|---|---|---|
| `reward` | `10.0` or `0.0` | The training signal: `10 * won`, the same scale as verl-agent. |
| `trajectory/accuracy` | `1.0` or `0.0` | Task success, logged as the per-trajectory success rate. |

The 0.1 invalid-action penalty is not part of this reward: the GiGPO estimator applies it per step
(`invalid_action_penalty_coef`).

## Usage

Pass the reward to an agent with `reward_fn=alfworld_reward`, or by name in a training config
(`agent.init_config.reward_name=alfworld_reward`). The agent calls it after each episode with the
episode's `context`, so it reads the same environment that `alfworld_step` acted on.

For a complete setup (an `ActionAgent` that writes `<action>...</action>`, the `alfworld_step`
tool and this reward), see
[`examples/train_scripts/alfworld/train_alfworld.sh`](https://github.com/Agent-One-Lab/AgentFly/blob/main/examples/train_scripts/alfworld/train_alfworld.sh).

Called directly, it takes only the context:

```python
result = await alfworld_reward(context=context)
print(result)  # {"reward": 10.0, "trajectory/accuracy": 1.0} if the task was completed
```

## Resource Management

The reward acquires the ALFWorld environment through `ALFWorldSpec` and
`context.acquire_resource(...)`, with the same scope as the tools, so both use the rollout's
environment. The maximum number of concurrent environments is `ALFWorldSpec.max_global_num`
(currently `32`).
