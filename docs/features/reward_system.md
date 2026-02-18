# Reward System

We put reward calculation into the agent side instead of trainer side and use a separate *reward* layer for it. This is for severl reasons:
1. Reward calculation is related to the task itself. Different tasks may need different rewards.
2. Reward calculation can be designed to be asynchronous for efficiency.

### Definition
Similar to tools, we can decide whether to use environments in the reward definition. The return should either be a value, or a dictionary containing `reward` as one of keys. We can use decorator `@tool` or inherit from the `BaseReward` class. Any additional keys in the returned dict (e.g. `em`, `f1`, `fmt`) are passed through and documented in training and validation.



```python
@reward(name="qa_f1_reward")
def qa_f1_reward(final_response: str, golden_answer: str, trajectory: List[str]) -> float:
    """A reward function that uses f1 score as reward value"""
    f1, precision, recall = f1_score(final_response, golden_answer)
    em = em_score(final_response, golden_answer)

    return {
        "reward": f1,
        "f1": f1,
        "em": em,
        "precision": precision,
        "recall": recall,
    }

class APIReward(BaseReward):
    name="api_reward"
    def __init__(self, api_key):
        self.api_key = api_key

    def call(query: str):

        # call request with api key
        result = requests.request(api_key=self.api_key, query=query)

        return result['reward']

@reward(name="webshop_reward", env_cls=WebAgentTextEnv, pool_size=8)
async def webshop_reward(final_response: str, env: WebAgentTextEnv, task_id: int) -> dict:
    """
    Calculates the reward for the WebShop environment based on the environment state. Match the purchased product with the golden answer characteristics.
    Actual logic for reward calculation is in the environment and partially in step method of the environment.
    Adapted from https://arxiv.org/pdf/2207.01206

    Args:
        final_response (str): The agent's predicted action or response. Not used in this reward function.
        env (WebAgentTextEnv): The environment instance for the WebShop task.
        task_id (int): The identifier for the current task. Used to match with golden answer.

    Returns:
        dict: A dictionary containing the reward (float) and output (str) from the environment step. If an error occurs, returns a reward of 0.0 and an error message as output.
    """
    try:
        result = await env.step('get_reward', task_id)
        return {
            "reward": result["reward"],
            "output": result["observation"],
        }
    except Exception as e:
        return {
            "reward": 0.0,
            "output": f"Error webshop reward function: {e}",
        }
```


## Predefined Fields

When agent uses the reward function, it will detects automatically for three keys: `final_response`, `trajectory`, and `id` and assign these values to rewards.

- `final_response`: The final response the agent generate for the task.
- `trajectory`: The whole interaction trajectory.
- `id`: The trajectory id. This will be the within the trajectory. And tools and rewards will share the same id for the same task. So tools and rewards are assigned same environments.

When defining the function, you can set these (except `id`) to your arguments and directly use them in you reward calculation. For `id`, you can give `env` as argument and directly use it. The chain rollout will ensure the reward give the same environment as in tools for the same task.

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

Each a `float` value or a dictionary containing `reward` as key should be returned. If the return value is `float`, it is directly used as rewards. If a dictionary is returned, the `reward` is used as rewards. While other keys are still documented.

Extra keys (besides `reward`) are logged as `reward_extra/{key}/mean`, `reward_extra/{key}/max`, `reward_extra/{key}/min` in the metrics produced by `compute_data_metrics` (`verl/verl/trainer/ppo/metric_utils.py`).