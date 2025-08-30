.. _code_tools:

Code Tools
==========

The Code environment provides tools for executing Python code in secure, isolated containers. These tools enable agents to run code snippets, perform calculations, data analysis, and other computational tasks safely.

Tools Reference
---------------

.. currentmodule:: agentfly.tools.src.code.tools

code_interpreter
~~~~~~~~~~~~~~~~

.. autofunction:: agentfly.tools.src.code.tools.code_interpreter

**Function Signature:**

.. code-block:: python

    async def code_interpreter(code: str, env: PythonSandboxEnv) -> str

**Description:** Execute Python code in a secure Docker container sandbox and return the output from stdout or stderr

**Parameters:**
    - **code** (str): The Python code to execute in the sandbox environment
    - **env** (PythonSandboxEnv): The Python sandbox environment instance for code execution

**Returns:**
    - **str**: The output from code execution (stdout) or error messages (stderr)

**Tool Configuration:**
    - **Environment Class**: ``PythonSandboxEnv``
    - **Pool Size**: 32 concurrent instances
    - **Stateful**: True (maintains state between calls)
    - **Description**: "Run the code in docker container and return the output from stdout or stderr"

Usage Examples
--------------

Basic Code Execution
~~~~~~~~~~~~~~~~~~~~

Execute simple Python expressions and statements:

.. code-block:: python

   from agentfly.tools import code_interpreter
   from agentfly.envs.python_env import PythonSandboxEnv
   
   # Create environment
   env = await PythonSandboxEnv.acquire()
   
   # Basic calculations
   result = await code_interpreter(
       code="print(2 + 2 * 3)",
       env=env
   )
   # Output: "8"
   
   # Variable assignment and use
   await code_interpreter(
       code="x = 42\ny = x * 2",
       env=env
   )
   
   result = await code_interpreter(
       code="print(f'Result: {y}')",
       env=env
   )
   # Output: "Result: 84"

Data Analysis and Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use standard Python libraries for data processing:

.. code-block:: python

   # Import and use libraries
   code = '''
   import math
   import json
   
   data = [1, 2, 3, 4, 5]
   mean = sum(data) / len(data)
   std_dev = math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))
   
   result = {
       "data": data,
       "mean": mean,
       "std_dev": std_dev
   }
   
   print(json.dumps(result, indent=2))
   '''
   
   result = await code_interpreter(code=code, env=env)

Error Handling
~~~~~~~~~~~~~~

The tool gracefully handles errors and exceptions:

.. code-block:: python

   # Syntax error example
   result = await code_interpreter(
       code="print('Hello World'",  # Missing closing parenthesis
       env=env
   )
   # Returns error message with details
   
   # Runtime error example
   result = await code_interpreter(
       code="print(undefined_variable)",
       env=env
   )
   # Returns: "NameError: name 'undefined_variable' is not defined"

State Persistence
~~~~~~~~~~~~~~~~~

Variables and imports persist within the same environment session:

.. code-block:: python

   # Define functions and variables
   await code_interpreter(
       code='''
       def fibonacci(n):
           if n <= 1:
               return n
           return fibonacci(n-1) + fibonacci(n-2)
       
       # Cache some results
       fib_cache = {i: fibonacci(i) for i in range(10)}
       ''',
       env=env
   )
   
   # Use previously defined function and data
   result = await code_interpreter(
       code="print([fib_cache[i] for i in range(5)])",
       env=env
   )
   # Output: "[0, 1, 1, 2, 3]"

File Operations
~~~~~~~~~~~~~~~

Work with files within the sandbox:

.. code-block:: python

   # Create and write to files
   await code_interpreter(
       code='''
       with open('data.txt', 'w') as f:
           f.write('Hello, World!\\nLine 2\\nLine 3')
       ''',
       env=env
   )
   
   # Read and process files
   result = await code_interpreter(
       code='''
       with open('data.txt', 'r') as f:
           lines = f.readlines()
       
       print(f"File has {len(lines)} lines")
       for i, line in enumerate(lines, 1):
           print(f"Line {i}: {line.strip()}")
       ''',
       env=env
   )

Integration with ReactAgent
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real-world usage with ReactAgent for problem-solving:

.. code-block:: python

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
       backend="async_vllm",
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

