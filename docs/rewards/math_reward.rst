.. _math_reward:

Math Reward Functions
=====================

.. currentmodule:: agentfly.rewards.math_reward

Math reward functions evaluate agent performance on mathematical problem-solving tasks with various behavioral requirements.

math_reward
-----------

.. autofunction:: math_reward

**Function Signature:**

.. code-block:: python

    def math_reward(prediction: str, golden_answer: str, **kwargs) -> float

**Description:** Basic mathematical answer correctness evaluation using symbolic math comparison.

**Parameters:**
    - **prediction** (str): Agent's predicted answer
    - **golden_answer** (str): Correct mathematical answer
    - **kwargs**: Additional arguments (ignored)

**Returns:**
    float: 1.0 if answers are mathematically equivalent, 0.0 otherwise

math_reward_tool
----------------

.. autofunction:: math_reward_tool

**Function Signature:**

.. code-block:: python

    def math_reward_tool(prediction: str, answer: str, trajectory: List[Dict]) -> dict

**Description:** Evaluates mathematical correctness with tool usage requirement.

**Parameters:**
    - **prediction** (str): Agent's predicted answer
    - **answer** (str): Correct mathematical answer  
    - **trajectory** (List[Dict]): Agent's conversation trajectory

**Returns:**
    dict: Dictionary containing:
        - **reward** (float): 
            - 0.0 if no tool used
            - 0.1 if tool used but answer incorrect
            - 1.0 if tool used and answer correct
        - **acc** (float): 1.0 if answer correct, 0.0 otherwise

math_reward_thought
-------------------

.. autofunction:: math_reward_think

**Function Signature:**

.. code-block:: python

    def math_reward_think(prediction: str, answer: str, trajectory: List[Dict]) -> dict

**Description:** Rewards mathematical correctness with thinking process requirement.

**Parameters:**
    - **prediction** (str): Agent's predicted answer
    - **answer** (str): Correct mathematical answer
    - **trajectory** (List[Dict]): Agent's conversation trajectory

**Returns:**
    dict: Dictionary containing:
        - **reward** (float):
            - 0.0 if no tool used or missing thought patterns
            - 0.1 if partial compliance (tool XOR thought)
            - 1.0 if both tool used and all responses start with "thought"
        - **acc** (float): 1.0 if answer correct, 0.0 otherwise

math_reward_think
-----------------

.. autofunction:: math_reward_think

**Function Signature:**

.. code-block:: python

    def math_reward_think(prediction: str, answer: str, trajectory: List[Dict]) -> dict

**Description:** Advanced reward requiring structured thinking in `<think>...</think>` tags.

**Parameters:**
    - **prediction** (str): Agent's predicted answer
    - **answer** (str): Correct mathematical answer
    - **trajectory** (List[Dict]): Agent's conversation trajectory

**Returns:**
    dict: Dictionary containing:
        - **reward** (float):
            - 0.0 if no `<think>` tags or no tool usage
            - 0.1 if requirements met but answer incorrect
            - 1.0 if requirements met and answer correct
        - **acc** (float): 1.0 if answer correct, 0.0 otherwise

Technical Details
-----------------

**Symbolic Math Comparison:**
    - Uses sympy for mathematical equivalence checking
    - Handles LaTeX expressions via `parse_latex`
    - Supports boxed answers: `\\boxed{expression}`
    - Robust to formatting differences

**Trajectory Analysis:**
    - Parses conversation messages by role
    - Detects tool usage through "tool" role messages
    - Analyzes assistant responses for format compliance
    - Supports both string and list content formats

**Think Tag Parsing:**
    - Extracts content from `<think>...</think>` tags
    - Handles incomplete or malformed thinking patterns
    - Requires both thinking and answer components

**Common Usage Patterns:**

.. code-block:: python

    # Basic correctness
    result = math_reward("\\boxed{42}", "\\boxed{42}")
    print(result)  # 1.0
    
    # Tool usage requirement
    trajectory = [
        {"role": "assistant", "content": "I need to calculate..."},
        {"role": "tool", "content": "calculation result"},
        {"role": "assistant", "content": "The answer is \\boxed{42}"}
    ]
    result = math_reward_tool("\\boxed{42}", "\\boxed{42}", trajectory)
    print(result)  # {"reward": 1.0, "acc": 1.0}
    
    # Structured thinking requirement
    trajectory = [
        {"role": "assistant", "content": "<think>Let me solve this step by step</think><answer>\\boxed{42}</answer>"},
        {"role": "tool", "content": "verification"},
    ]
    result = math_reward_think("\\boxed{42}", "\\boxed{42}", trajectory)
    print(result)  # {"reward": 1.0, "acc": 1.0}

**Use Cases:**
    - Mathematical problem solving evaluation
    - Process-based reward for tool usage
    - Training structured reasoning behaviors
    - Multi-step mathematical reasoning assessment 