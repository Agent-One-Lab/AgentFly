# Code Execution Reward

::: agentfly.rewards.code_reward.code_reward_test
    options:
      show_source: true

## Function Signature

```python
async def code_reward_test(prediction: str, context: Context) -> dict
```

## Description

Evaluates code execution in a sandboxed Python environment, providing binary success/failure feedback.

**Parameters:**
- **prediction** (str): Python code to execute.
- **context** (`Context`): Rollout execution context used to acquire the Python sandbox resource.

**Returns:**
dict: Dictionary containing:
- **reward** (float): 1.0 if execution successful, 0.0 if error occurred.
- **output** (str): Execution result or error message.

**Decorator Configuration:**
- **name**: `"code_reward_test"`
- **resource_spec**: `PythonSandboxSpec`
- **backend**: `"local"`

## Technical Details

**Implementation:**
- Executes code in isolated Python sandbox
- Captures both successful outputs and exceptions
- Returns binary reward based on execution success
- Provides detailed output for debugging

**Error Handling:**
- Catches all exceptions during code execution
- Returns error details in output field
- Ensures safe evaluation without affecting host system

**Example Usage:**

```python
from agentfly.core import Context
from agentfly.rewards.code_reward import code_reward_test

# Inside a rollout, `context` is injected and passed through to rewards.
result = await code_reward_test(
    prediction="print('Hello, World!')",
    context=context,
)
print(result)
# {"reward": 1.0, "output": "Hello, World!"}

result = await code_reward_test(
    prediction="print(undefined_variable)",
    context=context,
)
print(result)
# {"reward": 0.0, "output": "NameError: name 'undefined_variable' is not defined"}
```

**Use Cases:**
- Code generation evaluation
- Programming task assessment
- Syntax and runtime error detection
- Training code-writing agents

**Environment Integration:**
- Requires active Python sandbox environment
- Isolated execution prevents system interference
- Supports concurrent code evaluation through pooling
