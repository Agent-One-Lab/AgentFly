# Code Rewards

The Code reward system provides evaluation functions for code execution tasks. The rewards are typically binary and outcome-based, focusing on successful code execution and completion rather than incremental progress.

## Reward Functions Reference

### code_reward_test

::: agentfly.rewards.code_reward.code_reward_test
    options:
      show_source: true

**Function Signature:**

```python
async def code_reward_test(prediction: str, env: PythonSandboxEnv) -> Dict[str, Any]
```

**Description:** Evaluate the reward for code execution based on successful completion

**Parameters:**
- **prediction** (str): The code snippet to execute and evaluate
- **env** (PythonSandboxEnv): The Python sandbox environment instance for code execution

**Returns:**
- **Dict[str, Any]**: Dictionary containing:
    - **reward** (float): 1.0 for successful execution, 0.0 for failures
    - **output** (str): The execution result or error message

**Reward Configuration:**
- **Environment Class**: ``PythonSandboxEnv``
- **Pool Size**: 16 concurrent instances
- **Name**: "code_reward_test"

## Reward Structure

### Binary Reward System

The code reward system implements a simple binary evaluation:

* **Success (1.0)**: Code executes without throwing exceptions
* **Failure (0.0)**: Code execution results in errors or exceptions

This outcome-based approach focuses on:

* **Correctness**: Whether the code runs successfully
* **Completion**: Whether the execution finishes without errors
* **Binary Evaluation**: Clear success/failure distinction

## Usage Examples

### Basic Reward Evaluation

Evaluate simple code snippets:

```python
from agentfly.rewards.code_reward import code_reward_test
from agentfly.envs.python_env import PythonSandboxEnv

# Create environment
env = await PythonSandboxEnv.acquire()

# Successful code execution
result = await code_reward_test(
    prediction="print('Hello, World!')",
    env=env
)
# Returns: {"reward": 1.0, "output": "Hello, World!\n"}

# Failed code execution
result = await code_reward_test(
    prediction="print('Hello World'",  # Syntax error
    env=env
)
# Returns: {"reward": 0.0, "output": "SyntaxError: ..."}
```

### Custom Reward Functions

Create specialized reward functions for specific tasks:

```python
from agentfly.rewards.reward_base import reward

@reward(name="math_code_reward", env_cls=PythonSandboxEnv, pool_size=8)
async def math_code_reward(prediction: str, env: PythonSandboxEnv) -> dict:
    """Reward function for mathematical computation tasks"""
    try:
        result = await env.step(prediction)

        # Check if output contains expected mathematical result
        if "3.14159" in result or "math.pi" in prediction:
            return {"reward": 1.0, "output": result}
        elif any(keyword in prediction.lower() for keyword in ["math", "calculate", "compute"]):
            return {"reward": 0.5, "output": result}  # Partial credit
        else:
            return {"reward": 0.0, "output": result}

    except Exception as e:
        return {"reward": 0.0, "output": str(e)}
```
