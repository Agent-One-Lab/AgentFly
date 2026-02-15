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


def _extract_answer_tag(text: str) -> str:
    """Extract content between <answer> and </answer>, or return original if not present."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _format_ok(final_response: str, trajectory: List) -> tuple:
    """True if final_response has <answer>...</answer>, trajectory has tool calling, and all assistant turns except the last have <think>/</think>."""
    has_answer_tags = "<answer>" in final_response and "</answer>" in final_response
    
    has_tool_calling = any(
        isinstance(msg, dict) and msg.get("role") == "tool" for msg in trajectory
    )

    tool_call_count = sum(1 for msg in trajectory if msg.get("role") == "tool")

    # Collect assistant turns; only previous (non-last) ones must have think
    assistant_turns = []
    for msg in trajectory:
        if isinstance(msg, dict):
            if msg.get("role") == "assistant":
                content = msg.get("content") or msg.get("text") or ""
                assistant_turns.append(content)
            elif msg.get("role") == "tool":
                pass  # already counted has_tool_calling
        else:
            assistant_turns.append(str(msg))
    if not assistant_turns:
        previous_have_think = True
    else:
        previous = assistant_turns[:-1]  # all but last
        previous_have_think = all(
            "<think>" in c and "</think>" in c for c in previous if c
        )

    has_tool_call_answer = False
    answer = None
    last_message = trajectory[-1]
    if last_message["role"] == "assistant":
        if "tool_calls" in last_message and len(last_message["tool_calls"]) > 0:
            tool_call = last_message["tool_calls"][0]
            if "answer" in tool_call["function"]["arguments"]:
                has_tool_call_answer = True
                answer = tool_call["function"]["arguments"]["answer"]

    format_dict = {
        "has_think": previous_have_think,
        "has_answer_tags": has_answer_tags,
        "has_tool_calling": has_tool_calling,
        "tool_call_count": tool_call_count,
        "has_tool_call_answer": has_tool_call_answer,
        "answer": answer,
    }
    return format_dict


@reward(name="qa_em_format_reward")
def qa_em_format_reward(final_response: str, golden_answers: List[str], trajectory: List[str]) -> float:
    """
    Calculate the reward for the agent's response based on the EM score.

    - 1.0 if the format is correct, and the em is true
    - 0.1 if the format is correct, but the em is wrong
    - 0.0 if the format is incorrect
    """
    predicted = _extract_answer_tag(final_response)
    if not golden_answers:
        max_em, max_f1 = 0.0, 0.0
    else:
        max_em = max(em_score(predicted, g) for g in golden_answers)
        max_f1 = max(f1_score(predicted, g)[0] for g in golden_answers)
    format_dict = _format_ok(final_response, trajectory)

    reward = 0.0
    if format_dict["format"] and max_em:
        reward = 1.0
    elif format_dict["format"] and not max_em:
        reward = 0.1
    elif max_em and not format_dict["format"]:
        reward = 0.0

    return {
        "reward": reward,
        "em": max_em,
        "f1": max_f1,
        "fmt": 1.0 if format_dict["has_think"] and format_dict["has_answer_tags"] else 0.0,
        "fmt_think": 1.0 if format_dict["has_think"] else 0.0,
        "fmt_answer_tags": 1.0 if format_dict["has_answer_tags"] else 0.0,
        "fmt_tool": 1.0 if format_dict["has_tool_calling"] else 0.0,
    }

{
    "question": "...",
    "golden_answers": ["..."],
}

@reward(name="qa_em_reward")
def qa_em_reward(final_response: str, golden_answers: List[str], trajectory: List[str]) -> float:
    """
    Calculate the reward for the agent's response based on the EM score.

    - 1.0 if the format is correct, and the em is true
    - 0.1 if the format is correct, but the em is wrong
    - 0.0 if the format is incorrect
    """
    predicted = _extract_answer_tag(final_response)
    if not golden_answers:
        max_em, max_f1 = 0.0, 0.0
    else:
        max_em = max(em_score(predicted, g) for g in golden_answers)
        max_f1 = max(f1_score(predicted, g)[0] for g in golden_answers)
    format_dict = _format_ok(final_response, trajectory)

    reward = max_em

    return {
        "reward": reward,
        "em": max_em,
        "f1": max_f1,
        "fmt": 1.0 if format_dict["has_think"] and format_dict["has_answer_tags"] else 0.0,
        "fmt_think": 1.0 if format_dict["has_think"] else 0.0,
        "fmt_answer_tags": 1.0 if format_dict["has_answer_tags"] else 0.0,
        "fmt_tool": 1.0 if format_dict["has_tool_calling"] else 0.0,
    }


@reward(name="qa_em_reward_tool")
def qa_em_reward_tool(final_response: str, golden_answers: List[str], trajectory: List[str]) -> float:
    """
    Calculate the reward for the agent's response based on the EM score.

    - 1.0 if the format is correct, and the em is true
    - 0.1 if the format is correct, but the em is wrong
    - 0.0 if the format is incorrect
    """

    format_dict = _format_ok(final_response, trajectory)

    answer = format_dict["answer"]
    if answer is None:
        predicted = final_response
    else:
        predicted = answer
        
    if not golden_answers:
        max_em, max_f1 = 0.0, 0.0
    else:
        max_em = max(em_score(predicted, g) for g in golden_answers)
        max_f1 = max(f1_score(predicted, g)[0] for g in golden_answers)

    reward = 0.0
    if format_dict['has_tool_calling'] and max_em:
        reward = 1.0
    elif format_dict['has_tool_calling'] and not max_em:
        reward = 0.1
    elif not format_dict['has_tool_calling']:
        reward = 0.0

    return {
        "reward": reward,
        "em": max_em,
        "f1": max_f1,
        "has_tool_calling": 1.0 if format_dict["has_tool_calling"] else 0.0,
        "has_tool_call_answer": 1.0 if format_dict["has_tool_call_answer"] else 0.0,
    }


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
