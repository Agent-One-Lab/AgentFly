# Reward System

We put reward calculation into the agent side instead of trainer side and use a separate *reward* layer for it. This is for severl reasons:
1. Reward calculation is related to the task itself. Different tasks may need different rewards.
2. Reward calculation can be designed to be asynchronous for efficiency.

### Definition
Similar to tools, reward functions can be **purely functional** (no external resources) or **stateful** (using environments/resources via `Context`). The return should either be a float value, or a dictionary containing `reward` as one of keys. We can use the `@reward` decorator or inherit from the `BaseReward` class. Any additional keys in the returned dict (e.g. `em`, `f1`, `fmt`) are passed through and documented in training and validation.



A purely functional reward decorated with `@reward`:

```python
--8<-- "src/agentfly/rewards/qa_reward.py:qa_f1_reward_example"
```

A class-based reward that holds external credentials or state, by inheriting from `BaseReward`:

```python
class APIReward(BaseReward):
    name = "api_reward"

    def __init__(self, api_key):
        self.api_key = api_key

    def call(self, query: str):
        # call request with api key
        result = requests.request(api_key=self.api_key, query=query)
        return result["reward"]
```

A stateful async reward that shares an environment with the corresponding tools through `Context`:

```python
from agentfly.core import Context
from agentfly.envs.webshop_text_env import WebShopSpec

--8<-- "src/agentfly/rewards/webshop_reward.py:webshop_reward_example"
```


## Predefined Fields

When an agent uses a reward function, it will automatically detect and fill several special fields when they appear in the reward signature: `final_response`, `trajectory`, and (via `Context`) the rollout-level identifiers.

- `final_response`: The final response the agent generates for the task.
- `trajectory`: The whole interaction trajectory.
- `context`: The rollout execution context, which knows the rollout id, group id and provides `acquire_resource` for sharing environments with tools.

When defining the function, you can set these to your arguments and directly use them in reward calculation. For stateful rewards that need to share an environment with tools, prefer taking a `context: Context` argument and using `context.acquire_resource(...)` with the same `ResourceSpec` as the corresponding tools.

## Additional Fields

Beside predefined fields, you can give additional fields in your task input. The input take the following format:

```python
task_messages = {
    "messages": [
        "role": "user", "content": "Search the information about AgentFly and write a short summary."
    ],
    "length_penalty": True,
    "max_length": 2048,
}

await agent.run(
    messages=task_messages,
    max_turns=4,
)
```

In this example, two additional fields `length_penalty` and `max_length` is defined in the input. And your reward function can be defined with these two fields. After the agent finished the task, it will put these values to the reward. For example,

```python
@reward(name="summary_reward_with_penalty")
def summary_reward(final_response, length_penalty, max_length):
    if length_penalty:
        if len(final_response) > max_length:
            return 0.0
        else:
            return 1.0
    else:
        return 1.0
```

## Return Values

A reward function returns either a `float` or a `dict` containing a `reward` key. When a `float` is returned, it is used directly. When a `dict` is returned, the value at `reward` is used as the scalar reward and every other key is logged as an extra metric (`reward_extra/{key}/mean`, `.../max`, `.../min`) in the metrics produced by `compute_data_metrics`.

Internally, the framework normalizes both shapes into a typed `RewardResult` at the boundary (`agentfly.rewards.calculate_reward`), exposing `.reward: float` and `.extras: Dict[str, Any]`. You don't need to construct `RewardResult` yourself — the conversion is automatic. The typed form lands on the trajectory as `Trajectory.reward` (the float) and `Trajectory.metrics` (the extras dict).