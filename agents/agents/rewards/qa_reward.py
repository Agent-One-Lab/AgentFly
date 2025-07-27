import re
import string
from collections import Counter
from typing import List
from .reward_base import reward


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = 0.0

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC, ZERO_METRIC, ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC, ZERO_METRIC, ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC, ZERO_METRIC, ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def em_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    return float(normalized_prediction == normalized_ground_truth)


@reward(name="qa_f1_reward")
def qa_f1_reward(prediction: str, answer: str, trajectory: List[str]) -> float:
    # Extract answer from agent's response
    response = prediction
    f1, precision, recall = f1_score(response, answer)
    em = em_score(response, answer)

    return {
        "reward": f1,
        "f1": f1,
        "em": em,
        "precision": precision,
        "recall": recall,
    }

@reward(name="qa_f1_reward_format")
def qa_f1_reward_format(prediction: str, answer: str, trajectory: List[str]) -> float:
    """
    Calculate the reward for the agent's response based on the F1 score and EM score.
    The reward is 0.0 if the agent has not called any tool.
    The reward is the F1 score if the agent has called a tool.
    """
    has_called_tool = False
    call_tool_count = 0
    for msg in trajectory:
        if msg["role"] == "tool":
            has_called_tool = True
            call_tool_count += 1
    
    rewards_dict = {}
    # Require at least two tool calls (since the last tool call is the answer)
    if call_tool_count <= 1:
        rewards_dict.update({
            "reward": 0.0,
            "f1": 0.0,
            "em": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        })
    elif call_tool_count > 1:
        f1, precision, recall = f1_score(prediction, answer)
        em = em_score(prediction, answer)
        rewards_dict.update({
            "reward": f1,
            "f1": f1,
            "em": em,
            "precision": precision,
            "recall": recall,
        })
    else:
        raise ValueError(f"Invalid prediction or trajectory for qa reward with format: Trajectory: {trajectory}")
    
    return rewards_dict
