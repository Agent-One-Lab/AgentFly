import re
import string
from collections import Counter
from typing import Dict, List, Union

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

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC, ZERO_METRIC, ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
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
def qa_f1_reward(final_response: str, answer: str, trajectory: List[str]) -> float:
    """
    Calculate the reward for the agent's response based on the F1 score.

    Args:
        prediction (str): The agent's predicted answer
        answer (str): The correct answer
        trajectory (List[str]): The agent's conversation trajectory

    Returns:
        dict: A dictionary containing the reward, F1 score, EM score, precision, and recall
            - reward (float): The reward value
            - f1 (float): The F1 score
            - em (float): The EM score
            - precision (float): The precision score
            - recall (float): The recall score
    """
    response = final_response
    f1, precision, recall = f1_score(response, answer)
    em = em_score(response, answer)

    return {
        "reward": f1,
        "f1": f1,
        "em": em,
        "precision": precision,
        "recall": recall,
    }


@reward(name="qa_f1_reward_tool")
def qa_f1_reward_tool(final_response: str, answer: str, trajectory: List[str]) -> float:
    """
    Calculate the reward for the agent's response based on the F1 score and EM score.
    - 0.0 if no tool used
    - 0.1 if tool used but answer incorrect
    - 1.0 if tool used and answer correct

    Args:
        prediction (str): The agent's predicted answer
        answer (str): The correct answer
        trajectory (List[str]): The agent's conversation trajectory

    Returns:
        dict: A dictionary containing the reward, F1 score, EM score, precision, and recall
            - reward (float): The reward value
            - f1 (float): The F1 score
            - em (float): The EM score
            - precision (float): The precision score
            - recall (float): The recall score
    """
    # has_called_tool = False
    call_tool_count = 0
    for msg in trajectory:
        if msg["role"] == "tool":
            # has_called_tool = True
            call_tool_count += 1

    rewards_dict = {}
    # Require at least two tool calls (since the last tool call is the answer)
    if call_tool_count <= 1:
        rewards_dict.update(
            {
                "reward": 0.0,
                "f1": 0.0,
                "em": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
        )
    elif call_tool_count > 1:
        f1, precision, recall = f1_score(final_response, answer)
        em = em_score(final_response, answer)
        rewards_dict.update(
            {
                "reward": f1,
                "f1": f1,
                "em": em,
                "precision": precision,
                "recall": recall,
            }
        )
    else:
        raise ValueError(
            f"Invalid prediction or trajectory for qa reward with format: Trajectory: {trajectory}"
        )

    return rewards_dict


@reward(name="ok_vqa_reward")
def ok_vqa_reward(
    final_response: str, answers: List[str], trajectory: List[str]
) -> float:
    """
    Calculate the reward for the agent's response based on the F1 score and EM score.
    The reward is 0.0 if the agent has not called any tool.
    The reward is the F1 score if the agent has called a tool.
    """
    f1_scores = []
    for answer in answers:
        f1, precision, recall = f1_score(final_response, answer)
        f1_scores.append(f1)
    # All answers are the correct answer, take the max f1 score
    return max(f1_scores)


@reward(name="infoseek_reward")
def infoseek_reward(
    final_response: str,
    answer: Union[str, List[str]],
    answer_eval: List[str | Dict],
    trajectory: List[str],
) -> float:
    # format reward
    call_tool_count = 0
    for msg in trajectory:
        if msg["role"] == "tool":
            call_tool_count += 1

    f1_scores = []
    answers = []
    if isinstance(answer, str):
        answers.append(answer)
    elif isinstance(answer, list):
        answers.extend(answer)

    if isinstance(answer_eval[0], str):
        answers.extend(answer_eval)

    for _answer in answers:
        f1, precision, recall = f1_score(final_response, _answer)
        f1_scores.append(f1)

    max_f1_score = max(f1_scores)

    call_tool_reward = 1.0 if call_tool_count > 1 else 0.0

    reward = 0.2 * call_tool_reward + 0.8 * max_f1_score

    return reward
