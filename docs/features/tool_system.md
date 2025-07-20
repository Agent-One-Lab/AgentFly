## Tool System

### Unifying Interfaces with Tool Call

Tool calling are becoming stardards for LM agents. We use *tool* as an abstractive concept that represent all *functions*, *tools*, *APIs*, *environments*, etc. Therefore, agent rollout is unified as a repeatitive process of generation and tool calling. However, there are some challenges that we need to tackle for RL rollout:

1. Parallelism: To ensure efficiency, during the rollout we need to have multiple interactions in parallel. This requires the tool can be called in parallel.

2. Isolation: Some environments need to be isolated during the interaction (e.g. writing files in an os).


### Non-Stateful & Stateful Tool
We define two types of tools:

1. Non-Stateful Tool does not keep environment states. For such tool, we can simply write a function and decorate it with `@tool`.
```python
@tool(name="AdditionTool", description="Adds two numbers.")
    def add(a, b: int = 1):
        """
        Adds two numbers.

        Args:
            a (int): The first number.
            b (int): The second number which should be a non-negative integer.
        
        Returns:
            int: The sum of a and b.
        """
        return a + b
```

2. Stateful Tool keeps the environmental states. In definition, we need to specify an environment class (`env_cls`), and the pool size. The resource system will initialize the number of pool size environment instances. When calling the tool, it will require an instance from the central resource system, and will later release it back to the pool. Inside the tool, we can use the environment by calling the `step` method, which is generally the interface to interact with environment. For example, for `PythonSandboxEnv`, the `step` method accepts a code string, executing it and return anything on stdout or stderr.

```python
@tool(env_cls=PythonSandboxEnv, name="code_interpreter", description="Run the code in docker container and return the output from stdout or stderr", stateful=True, pool_size=16)
async def code_interpreter(code: str, env: PythonSandboxEnv):
    """
    Run the code in docker container and return the output from stdout or stderr
    Args:
        code (str): The code to run.
    Returns:
        str: The output from stdout or stderr
    """
    try:
        obs = await env.step(code)
        return obs
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"
```

### Tool Calling
When calling the tool, if it is a stateful tool, we need to specify an `id` argument for environment isolation. Calling with different `id`s are ensured to make use of different environment instances, while same `id` will use the same environment. A call with a new `id` will consume an environment instance in the pool, until it is released. **If there is no instance inside the pool, new call will be stuck until instances are released back to the pool.**

```python
result = code_interpreter("print('Hello World')", id="12345")
code_interpreter.release(id="12345")
```


