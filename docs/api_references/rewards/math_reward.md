# Math Reward Functions

Math reward functions evaluate agent performance on mathematical problem-solving tasks with various behavioral requirements.

## math_equal_reward

::: agentfly.rewards.math_equal_reward
    options:
      show_source: true

Returns 1.0 if the agent's answer is mathematically equivalent to the gold answer, 0.0 otherwise.

## math_equal_reward_tool

::: agentfly.rewards.math_equal_reward_tool
    options:
      show_source: true

Like `math_equal_reward` but gated on tool usage:

- 0.0 if no tool used
- 0.1 if tool used but answer incorrect
- 1.0 if tool used and answer correct

## math_equal_reward_think

::: agentfly.rewards.math_equal_reward_think
    options:
      show_source: true

Adds a structured-thinking requirement on top of `math_equal_reward_tool`: assistant turns must use `<think>...</think>` tags.

## Technical Details

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

```python
from agentfly.rewards import math_equal_reward, math_equal_reward_tool

# Basic correctness
--8<-- "tests/docs/rewards/test_reward_examples.py:math_equal_reward_basic"

# Tool usage requirement
--8<-- "tests/docs/rewards/test_reward_examples.py:math_equal_reward_tool_with_trajectory"
```

**Use Cases:**
- Mathematical problem solving evaluation
- Process-based reward for tool usage
- Training structured reasoning behaviors
- Multi-step mathematical reasoning assessment
