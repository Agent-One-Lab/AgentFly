.. _code_reward:

Code Execution Reward
======================

.. currentmodule:: agents.agents.rewards.code_reward

.. autofunction:: code_reward_test

Function Signature
------------------

.. code-block:: python

    async def code_reward_test(prediction: str, env: PythonSandboxEnv) -> dict

Description
-----------

Evaluates code execution in a sandboxed Python environment, providing binary success/failure feedback.

**Parameters:**
    - **prediction** (str): Python code to execute
    - **env** (PythonSandboxEnv): Python sandbox environment instance

**Returns:**
    dict: Dictionary containing:
        - **reward** (float): 1.0 if execution successful, 0.0 if error occurred
        - **output** (str): Execution result or error message

**Decorator Configuration:**
    - **name**: "code_reward_test"
    - **env_cls**: PythonSandboxEnv
    - **pool_size**: 16

Technical Details
-----------------

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

.. code-block:: python

    from agents.agents.rewards import get_reward_from_name
    from agents.agents.envs import PythonSandboxEnv
    
    # Get reward function
    reward_fn = get_reward_from_name("code_reward_test")
    
    # Create environment
    env = PythonSandboxEnv()
    await env.start()
    
    # Test successful code
    result = await reward_fn("print('Hello, World!')", env=env)
    print(result)
    # {"reward": 1.0, "output": "Hello, World!"}
    
    # Test erroneous code
    result = await reward_fn("print(undefined_variable)", env=env)
    print(result)
    # {"reward": 0.0, "output": "NameError: name 'undefined_variable' is not defined"}

**Use Cases:**
    - Code generation evaluation
    - Programming task assessment
    - Syntax and runtime error detection
    - Training code-writing agents

**Environment Integration:**
    - Requires active Python sandbox environment
    - Isolated execution prevents system interference
    - Supports concurrent code evaluation through pooling 