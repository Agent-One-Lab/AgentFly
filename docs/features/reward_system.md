## Reward System

Similar to tools, we can decide whether to use environments in the reward definition. The return should either be a value, or a dictionary containing `reward` as one of keys. 

```python
@reward(name="qa_f1_reward")
def qa_f1_reward(prediction: str, golden_answer: str, trajectory: List[str]) -> float:
    """A reward function that uses f1 score as reward value"""
    response = prediction
    f1, precision, recall = f1_score(response, golden_answer)
    em = em_score(response, golden_answer)

    return {
        "reward": f1,
        "f1": f1,
        "em": em,
        "precision": precision,
        "recall": recall,
    }

@reward(name="webshop_reward", env_cls=WebAgentTextEnv, pool_size=8)
async def webshop_reward(prediction: str, env: WebAgentTextEnv, task_id: int) -> dict:
    """
    Calculates the reward for the WebShop environment based on the environment state. Match the purchased product with the golden answer characteristics.
    Actual logic for reward calculation is in the environment and partially in step method of the environment.
    Adapted from https://arxiv.org/pdf/2207.01206

    Args:
        prediction (str): The agent's predicted action or response. Not used in this reward function.
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

