# Tutorial 2: Defining Custom Rewards

This tutorial covers how to define custom reward functions for training your AgentFly agents. Reward functions provide the feedback signal that guides the reinforcement learning process.

## Table of Contents

1. [Understanding Rewards](#understanding-rewards)
2. [Basic Reward Functions](#basic-reward-functions)
3. [Environment-Based Rewards](#environment-based-rewards)
4. [Trajectory-Aware Rewards](#trajectory-aware-rewards)
5. [Advanced Reward Patterns](#advanced-reward-patterns)
6. [LLM-as-Judge Rewards](#llm-as-judge-rewards)
7. [Best Practices](#best-practices)

## Understanding Rewards

Reward functions in AgentFly evaluate agent performance and provide scalar feedback for training. They can:

- Evaluate final predictions against ground truth
- Assess the quality of the reasoning process
- Analyze tool usage patterns
- Provide dense rewards throughout the trajectory

All reward functions must:
- Be decorated with `@reward`
- Accept `prediction` as the first parameter
- Return either a float or a dictionary containing a `reward` key

## Basic Reward Functions

### Exact Match Reward

```python
from agents.rewards.reward_base import reward

@reward(name="exact_match_reward")
def exact_match_reward(prediction: str, answer: str) -> dict:
    """
    Simple exact match reward function.
    
    Args:
        prediction (str): The agent's prediction
        answer (str): The correct answer
    
    Returns:
        dict: Reward dictionary with score and metadata
    """
    # Clean and normalize both strings
    pred_clean = prediction.strip().lower()
    answer_clean = answer.strip().lower()
    
    # Calculate exact match
    exact_match = 1.0 if pred_clean == answer_clean else 0.0
    
    return {
        "reward": exact_match,
        "exact_match": exact_match,
        "prediction_length": len(prediction),
        "answer_length": len(answer)
    }
```

### F1 Score Reward

```python
from sklearn.metrics import f1_score
import re

@reward(name="f1_reward")
def f1_reward(prediction: str, answer: str) -> dict:
    """
    F1-score based reward for text overlap.
    
    Args:
        prediction (str): The agent's prediction
        answer (str): The correct answer
    
    Returns:
        dict: Reward dictionary with F1 components
    """
    def tokenize(text):
        """Simple tokenization by splitting on whitespace and punctuation."""
        return re.findall(r'\b\w+\b', text.lower())
    
    pred_tokens = set(tokenize(prediction))
    answer_tokens = set(tokenize(answer))
    
    if not answer_tokens:
        return {"reward": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    # Calculate precision, recall, and F1
    intersection = pred_tokens & answer_tokens
    
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(answer_tokens) if answer_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "reward": f1,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "token_overlap": len(intersection)
    }
```

### Math Problem Reward

```python
import re

@reward(name="math_reward")
def math_reward(prediction: str, answer: str) -> dict:
    """
    Reward function for mathematical problems.
    
    Args:
        prediction (str): The agent's prediction
        answer (str): The correct answer
    
    Returns:
        dict: Reward with mathematical accuracy
    """
    def extract_boxed_answer(text):
        """Extract answer from \\boxed{} format."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        return matches[-1] if matches else text.strip()
    
    def normalize_number(text):
        """Normalize numerical answers."""
        try:
            # Try to convert to float for numerical comparison
            return float(text.replace(',', ''))
        except ValueError:
            # Return original text for non-numerical answers
            return text.strip().lower()
    
    # Extract answers
    pred_answer = extract_boxed_answer(prediction)
    true_answer = extract_boxed_answer(answer)
    
    # Normalize for comparison
    pred_normalized = normalize_number(pred_answer)
    true_normalized = normalize_number(true_answer)
    
    # Check for exact match
    exact_match = pred_normalized == true_normalized
    
    # For numerical answers, also check approximate equality
    approx_match = False
    if isinstance(pred_normalized, float) and isinstance(true_normalized, float):
        approx_match = abs(pred_normalized - true_normalized) < 1e-6
    
    reward_score = 1.0 if (exact_match or approx_match) else 0.0
    
    return {
        "reward": reward_score,
        "exact_match": exact_match,
        "approx_match": approx_match,
        "predicted_answer": pred_answer,
        "true_answer": true_answer
    }
```

## Environment-Based Rewards

For rewards that require stateful environments, use the `env_cls` parameter:

### Code Execution Reward

```python
from agents.envs.python_env import PythonSandboxEnv
from agents.rewards.reward_base import reward

@reward(
    name="code_execution_reward",
    env_cls=PythonSandboxEnv,
    pool_size=8
)
async def code_execution_reward(prediction: str, env: PythonSandboxEnv, test_cases: list) -> dict:
    """
    Reward based on code execution results.
    
    Args:
        prediction (str): The agent's code prediction
        env (PythonSandboxEnv): Code execution environment
        test_cases (list): List of test cases to run
    
    Returns:
        dict: Reward based on test case success rate
    """
    try:
        # Extract code from prediction (assuming it's in code blocks)
        import re
        code_pattern = r'```python\n(.*?)\n```'
        code_matches = re.findall(code_pattern, prediction, re.DOTALL)
        
        if not code_matches:
            return {
                "reward": 0.0,
                "error": "No Python code found in prediction",
                "tests_passed": 0,
                "total_tests": len(test_cases)
            }
        
        code = code_matches[0]
        
        # Run the code in the environment
        exec_result = await env.step(code)
        
        if "error" in exec_result:
            return {
                "reward": 0.0,
                "error": exec_result["error"],
                "tests_passed": 0,
                "total_tests": len(test_cases)
            }
        
        # Run test cases
        passed_tests = 0
        test_results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                test_result = await env.step(test_case)
                if "error" not in test_result:
                    passed_tests += 1
                    test_results.append({"test": i, "passed": True})
                else:
                    test_results.append({"test": i, "passed": False, "error": test_result["error"]})
            except Exception as e:
                test_results.append({"test": i, "passed": False, "error": str(e)})
        
        success_rate = passed_tests / len(test_cases) if test_cases else 0.0
        
        return {
            "reward": success_rate,
            "tests_passed": passed_tests,
            "total_tests": len(test_cases),
            "success_rate": success_rate,
            "test_results": test_results
        }
        
    except Exception as e:
        return {
            "reward": 0.0,
            "error": f"Execution error: {str(e)}",
            "tests_passed": 0,
            "total_tests": len(test_cases)
        }
```

### Web Environment Reward

```python
from agents.envs.web_env import WebEnv
from agents.rewards.reward_base import reward

@reward(
    name="web_task_reward",
    env_cls=WebEnv,
    pool_size=4
)
async def web_task_reward(prediction: str, env: WebEnv, target_url: str, success_criteria: dict) -> dict:
    """
    Reward for web-based tasks.
    
    Args:
        prediction (str): Agent's action sequence
        env (WebEnv): Web environment
        target_url (str): Target URL to navigate to
        success_criteria (dict): Criteria for task success
    
    Returns:
        dict: Reward based on web task completion
    """
    try:
        # Parse actions from prediction
        actions = parse_web_actions(prediction)
        
        # Execute actions in environment
        for action in actions:
            result = await env.step(action)
            if "error" in result:
                return {
                    "reward": 0.0,
                    "error": result["error"],
                    "actions_completed": actions.index(action)
                }
        
        # Check success criteria
        page_state = await env.step("get_page_state")
        success_score = evaluate_success_criteria(page_state, success_criteria)
        
        return {
            "reward": success_score,
            "success_score": success_score,
            "actions_completed": len(actions),
            "final_url": page_state.get("url", ""),
            "criteria_met": success_score > 0.8
        }
        
    except Exception as e:
        return {
            "reward": 0.0,
            "error": f"Web task error: {str(e)}"
        }

def parse_web_actions(prediction: str) -> list:
    """Parse web actions from agent prediction."""
    # Implementation to extract actions like click, type, navigate
    # This would be specific to your web action format
    pass

def evaluate_success_criteria(page_state: dict, criteria: dict) -> float:
    """Evaluate how well the page state meets success criteria."""
    # Implementation to check if criteria are met
    # Return a score between 0.0 and 1.0
    pass
```

## Trajectory-Aware Rewards

These rewards analyze the entire reasoning trajectory, not just the final answer:

### Process Quality Reward

```python
@reward(name="process_reward")
def process_reward(prediction: str, answer: str, trajectory: list) -> dict:
    """
    Reward that considers both outcome and reasoning process.
    
    Args:
        prediction (str): Final prediction
        answer (str): Correct answer
        trajectory (list): List of reasoning steps
    
    Returns:
        dict: Combined reward for outcome and process
    """
    # Outcome reward (final answer correctness)
    outcome_reward = 1.0 if prediction.strip().lower() == answer.strip().lower() else 0.0
    
    # Process analysis
    tool_usage_score = analyze_tool_usage(trajectory)
    reasoning_quality = analyze_reasoning_quality(trajectory)
    efficiency_score = analyze_efficiency(trajectory)
    
    # Combine scores
    process_score = (tool_usage_score + reasoning_quality + efficiency_score) / 3.0
    
    # Final reward combines outcome and process
    final_reward = 0.7 * outcome_reward + 0.3 * process_score
    
    return {
        "reward": final_reward,
        "outcome_reward": outcome_reward,
        "process_score": process_score,
        "tool_usage_score": tool_usage_score,
        "reasoning_quality": reasoning_quality,
        "efficiency_score": efficiency_score,
        "trajectory_length": len(trajectory)
    }

def analyze_tool_usage(trajectory: list) -> float:
    """Analyze quality of tool usage in trajectory."""
    if not trajectory:
        return 0.0
    
    tool_calls = [step for step in trajectory if step.get("role") == "tool"]
    
    # Reward appropriate tool usage
    tool_diversity = len(set(call.get("name", "") for call in tool_calls))
    usage_ratio = len(tool_calls) / len(trajectory)
    
    # Optimal usage ratio (not too many, not too few)
    optimal_ratio = 0.3
    ratio_score = 1.0 - abs(usage_ratio - optimal_ratio) / optimal_ratio
    
    return (tool_diversity / 5.0 + ratio_score) / 2.0

def analyze_reasoning_quality(trajectory: list) -> float:
    """Analyze quality of reasoning steps."""
    reasoning_steps = [step for step in trajectory 
                      if step.get("role") == "assistant" and "thought" in step.get("content", "").lower()]
    
    if not reasoning_steps:
        return 0.0
    
    # Check for logical progression, explanations, etc.
    quality_indicators = 0
    for step in reasoning_steps:
        content = step.get("content", "").lower()
        if any(word in content for word in ["because", "therefore", "since", "given"]):
            quality_indicators += 1
        if any(word in content for word in ["let's", "first", "next", "then"]):
            quality_indicators += 1
    
    return min(quality_indicators / (len(reasoning_steps) * 2), 1.0)

def analyze_efficiency(trajectory: list) -> float:
    """Analyze efficiency of the solution path."""
    # Penalize overly long trajectories
    length_penalty = max(0, 1.0 - (len(trajectory) - 5) * 0.1)
    
    # Reward direct progress toward solution
    # This would need to be task-specific
    
    return max(0.0, length_penalty)
```

### Multi-Step Reasoning Reward

```python
@reward(name="multi_step_reasoning_reward")
def multi_step_reasoning_reward(prediction: str, answer: str, trajectory: list, steps: list) -> dict:
    """
    Reward for multi-step reasoning problems.
    
    Args:
        prediction (str): Final answer
        answer (str): Correct answer
        trajectory (list): Agent's trajectory
        steps (list): Expected reasoning steps
    
    Returns:
        dict: Reward based on step-by-step correctness
    """
    # Check final answer
    final_correct = prediction.strip() == answer.strip()
    
    # Analyze intermediate steps
    step_scores = []
    trajectory_text = " ".join([step.get("content", "") for step in trajectory])
    
    for i, expected_step in enumerate(steps):
        # Check if this step appears in the trajectory
        step_present = check_step_present(trajectory_text, expected_step)
        step_scores.append(1.0 if step_present else 0.0)
    
    # Calculate progressive reward
    steps_completed = sum(step_scores)
    step_completion_rate = steps_completed / len(steps) if steps else 0.0
    
    # Combine final answer and step completion
    final_reward = 0.5 * (1.0 if final_correct else 0.0) + 0.5 * step_completion_rate
    
    return {
        "reward": final_reward,
        "final_correct": final_correct,
        "steps_completed": steps_completed,
        "total_steps": len(steps),
        "step_completion_rate": step_completion_rate,
        "step_scores": step_scores
    }

def check_step_present(trajectory_text: str, expected_step: str) -> bool:
    """Check if an expected reasoning step is present in trajectory."""
    # This would use fuzzy matching or semantic similarity
    # For simplicity, using keyword matching here
    keywords = expected_step.lower().split()
    return any(keyword in trajectory_text.lower() for keyword in keywords)
```

## LLM-as-Judge Rewards

Use another LLM to evaluate agent performance:

### LLM Judge Reward

```python
import openai
from agents.rewards.reward_base import reward

@reward(name="llm_judge_reward")
def llm_judge_reward(prediction: str, answer: str, question: str) -> dict:
    """
    Use an LLM as a judge to evaluate agent responses.
    
    Args:
        prediction (str): Agent's prediction
        answer (str): Reference answer
        question (str): Original question
    
    Returns:
        dict: LLM-based evaluation score
    """
    judge_prompt = f"""
You are an expert judge evaluating AI agent responses. Rate the quality of the prediction compared to the reference answer.

Question: {question}

Reference Answer: {answer}

Agent Prediction: {prediction}

Evaluate the prediction on:
1. Correctness (0-4 points): How accurate is the answer?
2. Completeness (0-3 points): Does it address all parts of the question?
3. Clarity (0-3 points): Is the explanation clear and well-structured?

Provide scores and a brief explanation. Format your response as:
Correctness: X/4
Completeness: X/3
Clarity: X/3
Total: X/10
Explanation: [brief explanation]
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1
        )
        
        judge_response = response.choices[0].message.content
        
        # Parse scores from response
        scores = parse_judge_scores(judge_response)
        total_score = scores.get("total", 0)
        normalized_score = total_score / 10.0  # Normalize to 0-1
        
        return {
            "reward": normalized_score,
            "llm_score": total_score,
            "correctness": scores.get("correctness", 0),
            "completeness": scores.get("completeness", 0),
            "clarity": scores.get("clarity", 0),
            "judge_explanation": scores.get("explanation", ""),
            "judge_response": judge_response
        }
        
    except Exception as e:
        # Fallback to simple exact match if LLM judge fails
        fallback_score = 1.0 if prediction.strip().lower() == answer.strip().lower() else 0.0
        return {
            "reward": fallback_score,
            "error": f"LLM judge failed: {str(e)}",
            "fallback_used": True
        }

def parse_judge_scores(response: str) -> dict:
    """Parse numerical scores from judge response."""
    import re
    
    scores = {}
    
    # Extract scores using regex
    correctness_match = re.search(r'Correctness:\s*(\d+)', response)
    completeness_match = re.search(r'Completeness:\s*(\d+)', response)
    clarity_match = re.search(r'Clarity:\s*(\d+)', response)
    total_match = re.search(r'Total:\s*(\d+)', response)
    
    if correctness_match:
        scores["correctness"] = int(correctness_match.group(1))
    if completeness_match:
        scores["completeness"] = int(completeness_match.group(1))
    if clarity_match:
        scores["clarity"] = int(clarity_match.group(1))
    if total_match:
        scores["total"] = int(total_match.group(1))
    
    # Extract explanation
    explanation_match = re.search(r'Explanation:\s*(.+)', response, re.DOTALL)
    if explanation_match:
        scores["explanation"] = explanation_match.group(1).strip()
    
    return scores
```

## Advanced Reward Patterns

### Composite Reward

```python
@reward(name="composite_reward")
def composite_reward(prediction: str, answer: str, trajectory: list, **kwargs) -> dict:
    """
    Composite reward combining multiple evaluation criteria.
    
    Args:
        prediction (str): Agent's prediction
        answer (str): Correct answer
        trajectory (list): Agent's trajectory
        **kwargs: Additional parameters
    
    Returns:
        dict: Composite reward with detailed breakdown
    """
    # Individual reward components
    accuracy_reward = exact_match_reward(prediction, answer)["reward"]
    f1_reward_score = f1_reward(prediction, answer)["reward"]
    process_reward_score = process_reward(prediction, answer, trajectory)["reward"]
    
    # Weighted combination
    weights = {"accuracy": 0.5, "f1": 0.2, "process": 0.3}
    
    composite_score = (
        weights["accuracy"] * accuracy_reward +
        weights["f1"] * f1_reward_score +
        weights["process"] * process_reward_score
    )
    
    return {
        "reward": composite_score,
        "accuracy_reward": accuracy_reward,
        "f1_reward": f1_reward_score,
        "process_reward": process_reward_score,
        "weights": weights
    }
```

### Adaptive Reward

```python
@reward(name="adaptive_reward")
def adaptive_reward(prediction: str, answer: str, difficulty: str = "medium") -> dict:
    """
    Adaptive reward that changes based on problem difficulty.
    
    Args:
        prediction (str): Agent's prediction
        answer (str): Correct answer
        difficulty (str): Problem difficulty level
    
    Returns:
        dict: Difficulty-adjusted reward
    """
    base_reward = exact_match_reward(prediction, answer)["reward"]
    
    # Adjust based on difficulty
    difficulty_multipliers = {
        "easy": 0.8,
        "medium": 1.0,
        "hard": 1.5,
        "expert": 2.0
    }
    
    multiplier = difficulty_multipliers.get(difficulty, 1.0)
    adjusted_reward = base_reward * multiplier
    
    # Apply sigmoid to keep rewards in reasonable range
    import math
    normalized_reward = 2 / (1 + math.exp(-adjusted_reward)) - 1
    
    return {
        "reward": normalized_reward,
        "base_reward": base_reward,
        "difficulty": difficulty,
        "multiplier": multiplier,
        "adjusted_reward": adjusted_reward
    }
```

## Best Practices

### 1. Return Consistent Format

Always return a dictionary with at least a `reward` key:

```python
@reward(name="good_reward")
def good_reward(prediction: str, answer: str) -> dict:
    """Good reward function with consistent format."""
    score = calculate_score(prediction, answer)
    return {
        "reward": score,
        "additional_info": "helpful metadata"
    }
```

### 2. Handle Edge Cases

```python
@reward(name="robust_reward")
def robust_reward(prediction: str, answer: str) -> dict:
    """Robust reward with edge case handling."""
    # Handle empty inputs
    if not prediction or not answer:
        return {"reward": 0.0, "error": "Empty input"}
    
    # Handle None inputs
    if prediction is None or answer is None:
        return {"reward": 0.0, "error": "None input"}
    
    # Normal processing
    score = calculate_score(prediction, answer)
    return {"reward": score}
```

### 3. Provide Meaningful Metadata

```python
@reward(name="informative_reward")
def informative_reward(prediction: str, answer: str) -> dict:
    """Reward with informative metadata."""
    score = calculate_score(prediction, answer)
    
    return {
        "reward": score,
        "prediction_length": len(prediction),
        "answer_length": len(answer),
        "confidence": get_confidence_score(prediction),
        "category": classify_response(prediction)
    }
```

### 4. Test Your Rewards

```python
def test_reward_function():
    """Test cases for reward functions."""
    # Test exact match
    result = exact_match_reward("correct", "correct")
    assert result["reward"] == 1.0
    
    # Test mismatch
    result = exact_match_reward("wrong", "correct")
    assert result["reward"] == 0.0
    
    # Test edge cases
    result = exact_match_reward("", "")
    assert "reward" in result
```

### 5. Scale Appropriately

Ensure rewards are on an appropriate scale (typically 0-1):

```python
@reward(name="scaled_reward")
def scaled_reward(prediction: str, answer: str) -> dict:
    """Properly scaled reward."""
    raw_score = calculate_raw_score(prediction, answer)
    
    # Scale to 0-1 range
    max_possible_score = get_max_score()
    scaled_score = min(raw_score / max_possible_score, 1.0)
    
    return {"reward": scaled_score}
```

## Usage in Training

Once defined, rewards are used in training configuration:

```python
# In your training script
from my_rewards import math_reward, process_reward

agent = ReactAgent(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    tools=[...],
    reward_fn=math_reward  # Use your custom reward
)
```

Or specified by name in configuration:

```bash
# In training command
python -m verl.trainer.main_ppo \
    agent.reward_name=math_reward \
    ...
```

## Next Steps

Now that you understand reward functions, proceed to:
- [Tutorial 3: Customizing Agents](03_customizing_agents.md) to learn about agent architecture
- [Tutorial 4: Data Preparation](04_data_preparation.md) to understand data formatting

For a complete example, see [Tutorial 7: Complete Pipeline](07_complete_pipeline.md).