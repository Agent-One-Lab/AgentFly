# Question Answering Rewards

QA reward functions evaluate agent performance on question answering tasks using F1 score and exact match metrics.

## qa_f1_reward

::: agentfly.rewards.qa_reward.qa_f1_reward
    options:
      show_source: true

**Returns** a dictionary with `reward` (F1 score), `f1`, `em`, `precision`, and `recall`.

## qa_f1_reward_tool

::: agentfly.rewards.qa_reward.qa_f1_reward_tool
    options:
      show_source: true

Same metrics as `qa_f1_reward` but the reward is gated on the agent having made at least one tool call (otherwise `reward` is 0.0).

## Technical Details

**Text Normalization:**
- Removes articles (a, an, the)
- Normalizes whitespace
- Removes punctuation
- Converts to lowercase

**F1 Score Calculation:**
- Token-based overlap between prediction and ground truth
- Precision = common_tokens / prediction_tokens
- Recall = common_tokens / ground_truth_tokens
- F1 = 2 * (precision * recall) / (precision + recall)

**Exact Match (EM):**
- Binary score: 1.0 if normalized answers are identical, 0.0 otherwise
- Special handling for yes/no/noanswer responses

**Tool Usage Detection:**
- Counts messages with "tool" role in trajectory
- `qa_f1_reward_tool` requires at least one tool call

**Example Usage:**

```python
from agentfly.rewards import qa_f1_reward, qa_f1_reward_tool

# Basic F1 evaluation
--8<-- "tests/docs/rewards/test_reward_examples.py:qa_f1_reward_basic"

# With tool usage requirement
--8<-- "tests/docs/rewards/test_reward_examples.py:qa_f1_reward_tool_with_trajectory"
```

**Special Cases:**
- Yes/No questions: Must match exactly or return 0.0
- Empty predictions: Return 0.0 for all metrics
- No token overlap: Return 0.0 for all metrics

**Use Cases:**
- Reading comprehension evaluation
- Information retrieval task assessment
- Knowledge-based question answering
- Tool-augmented QA system evaluation
