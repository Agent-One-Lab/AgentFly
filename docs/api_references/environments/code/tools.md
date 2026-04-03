# Code Tools

The Code environment provides tools for executing Python code in secure, isolated containers. These tools enable agents to run code snippets, perform calculations, data analysis, and other computational tasks safely.

## Tools Reference

### code_interpreter

::: agentfly.tools.src.code.tools.code_interpreter
    options:
      show_source: true

**Function Signature:**

```python
async def code_interpreter(code: str, context: Context) -> str
```

**Description:** Execute Python code in a sandboxed container environment acquired via `context.acquire_resource(...)`, and return the output from stdout or stderr.

**Parameters:**
- **code** (str): The Python code to execute in the sandbox environment
- **context** (`Context`): Injected rollout context. During agent rollouts, `Context` is provided automatically.

**Returns:**
- **str**: The output from code execution (stdout) or error messages (stderr)

**Tool Configuration:**
- **ResourceSpec**: `PythonSandboxSpec`
- **Resource scope**: `global` (the sandbox is released when the rollout ends)
- **Stateful behavior**: variable state persists within the same acquired sandbox instance.

## Usage Examples

### Usage with an Agent (recommended)

In AgentFly rollouts, `code_interpreter` is invoked via a tool call and receives `context: Context` automatically. The tool acquires a sandboxed environment via `PythonSandboxSpec` and returns stdout/stderr.

### Integration with ReactAgent

Real-world usage with ReactAgent for problem-solving:

```python
from agentfly.agents.react.react_agent import ReactAgent
from agentfly.rewards.code_reward import code_reward_test

# Task information for the agent
task_info = """Execute Python code to solve computational problems.
Use code_interpreter to run calculations, analysis, and data processing tasks."""

# Initialize ReactAgent with code_interpreter tool
react_agent = ReactAgent(
    "Qwen/Qwen2.5-7B-Instruct",
    tools=[code_interpreter],
    reward_fn=code_reward_test,
    template="qwen-chat",
    task_info=task_info,
    backend_config={"backend": "async_vllm"},
    debug=True
)

# Agent can now use code execution for calculations
await react_agent.run_async(
    max_steps=5,
    start_messages=[{
        "messages": [{"role": "user", "content": "Calculate the standard deviation of the numbers [1, 4, 6, 7, 12, 15, 18, 20] and explain the result"}],
        "question": "Calculate the standard deviation of the numbers [1, 4, 6, 7, 12, 15, 18, 20] and explain the result"
    }],
    num_chains=1
)
```
