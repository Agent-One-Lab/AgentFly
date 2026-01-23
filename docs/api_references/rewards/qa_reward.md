# Question Answering Rewards

QA reward functions evaluate agent performance on question answering tasks using F1 score and exact match metrics.

## qa_f1_reward

::: agentfly.rewards.qa_reward.qa_f1_reward
    options:
      show_source: true

**Function Signature:**

```python
def qa_f1_reward(prediction: str, golden_answer: str, trajectory: List[str]) -> dict
```

**Description:** Evaluates QA performance using F1 score and exact match metrics.

**Parameters:**
- **prediction** (str): Agent's predicted answer
- **golden_answer** (str): Correct answer
- **trajectory** (List[str]): Agent's conversation trajectory (unused in basic version)

**Returns:**
dict: Dictionary containing:
- **reward** (float): F1 score (0.0 to 1.0)
- **f1** (float): F1 score between prediction and golden answer
- **em** (float): Exact match score (0.0 or 1.0)
- **precision** (float): Precision score
- **recall** (float): Recall score

## qa_f1_reward_format

::: agentfly.rewards.qa_reward.qa_f1_reward_tool
    options:
      show_source: true

**Function Signature:**

```python
def qa_f1_reward_format(prediction: str, answer: str, trajectory: List[Dict]) -> dict
```

**Description:** F1-based QA evaluation with tool usage requirement.

**Parameters:**
- **prediction** (str): Agent's predicted answer
- **answer** (str): Correct answer
- **trajectory** (List[Dict]): Agent's conversation trajectory

**Returns:**
dict: Dictionary containing:
- **reward** (float): F1 score if tool used (≥2 tool calls), 0.0 otherwise
- **f1** (float): F1 score between prediction and answer
- **em** (float): Exact match score
- **precision** (float): Precision score
- **recall** (float): Recall score

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
- Requires minimum 2 tool calls (including final answer)
- Returns 0.0 reward if insufficient tool usage

**Example Usage:**

```python
# Basic F1 evaluation
result = qa_f1_reward("Paris is the capital", "Paris", [])
print(result)
# {"reward": 0.67, "f1": 0.67, "em": 0.0, "precision": 0.5, "recall": 1.0}

# With tool usage requirement
trajectory = [
    {"role": "assistant", "content": "I need to search for information"},
    {"role": "tool", "content": "search results"},
    {"role": "assistant", "content": "Based on my search, the answer is Paris"}
]
result = qa_f1_reward_format("Paris", "Paris", trajectory)
print(result)
# {"reward": 1.0, "f1": 1.0, "em": 1.0, "precision": 1.0, "recall": 1.0}
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

**Metric Interpretation:**
- **F1 Score**: Harmonic mean of precision and recall, good overall metric
- **Exact Match**: Strict evaluation requiring perfect answers
- **Precision**: Relevance of prediction tokens
- **Recall**: Coverage of ground truth tokens
