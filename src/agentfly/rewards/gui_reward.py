# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import logging
import re
from typing import Dict, List, Tuple

from ..utils.ui_action_parser import IMAGE_FACTOR, parse_action_to_structure_output
from .reward_base import reward

logger = logging.getLogger(__name__)

# Image dimensions for testing
TEST_IMAGE_HEIGHT = 1080
TEST_IMAGE_WIDTH = 1920


def normalize_answer(s: str) -> set:
    """Normalize answer string for comparison."""

    def remove_punctuation(text):
        return re.sub(r"[^\w\s]", "", text)

    def lower(text):
        return text.lower()

    return set(lower(remove_punctuation(s)).split())


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Calculate F1 score between prediction and ground truth."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if not normalized_prediction and not normalized_ground_truth:
        return 1.0, 1.0, 1.0

    common_tokens = normalized_prediction.intersection(normalized_ground_truth)

    precision = (
        len(common_tokens) / len(normalized_prediction)
        if normalized_prediction
        else 0.0
    )
    recall = (
        len(common_tokens) / len(normalized_ground_truth)
        if normalized_ground_truth
        else 0.0
    )

    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return f1, precision, recall


def extract_action(content: str) -> str:
    """Extract action type from response content."""
    try:
        parsed = parse_action_to_structure_output(
            content,
            IMAGE_FACTOR,
            TEST_IMAGE_HEIGHT,
            TEST_IMAGE_WIDTH,
            model_type="default",
        )
        if parsed and len(parsed) > 0:
            action_dict = parsed[0]
            action_type = action_dict.get("action_type", "no action")

            # Check for specific action types in raw text
            if action_type == "click" and "Action:" in content:
                action_text = content.split("Action:")[1].strip()
                if action_text.startswith("left_double"):
                    return "left_double"
                elif action_text.startswith("right_single"):
                    return "right_single"
                elif action_text.startswith("click"):
                    return "click"
                else:
                    return "click"

            # Map normalized action types
            if action_type == "hotkey":
                return "hotkey"
            elif action_type == "drag":
                return "drag"

            return action_type
    except Exception as e:
        logger.debug(f"[extract_action] Error: {e}")
    return "no action"


def extract_input_text(content: str) -> str:
    """Extract input text from action content."""
    try:
        parsed = parse_action_to_structure_output(
            content,
            IMAGE_FACTOR,
            TEST_IMAGE_HEIGHT,
            TEST_IMAGE_WIDTH,
            model_type="default",
        )
        if parsed and len(parsed) > 0:
            action_dict = parsed[0]
            action_type = action_dict.get("action_type")
            action_inputs = action_dict.get("action_inputs", {})

            # Extract text based on action type
            if action_type == "type":
                return action_inputs.get("content", "")
            elif action_type == "scroll":
                return action_inputs.get("direction", "down")
            elif action_type == "hotkey":
                return action_inputs.get("key", action_inputs.get("hotkey", ""))
            elif action_type == "finished":
                return action_inputs.get("content", "")

            return ""
    except Exception as e:
        logger.debug(f"[extract_input_text] Error: {e}")
    return ""


def extract_coord(content: str) -> Tuple[list, bool]:
    """Extract coordinates from action content and normalize them to 0-1 range."""
    try:
        parsed = parse_action_to_structure_output(
            content,
            IMAGE_FACTOR,
            TEST_IMAGE_HEIGHT,
            TEST_IMAGE_WIDTH,
            model_type="default",
        )
        if parsed and len(parsed) > 0:
            action_dict = parsed[0]
            action_inputs = action_dict.get("action_inputs", {})

            # Try to get coordinates from start_box
            if "start_box" in action_inputs:
                try:
                    coords = ast.literal_eval(action_inputs["start_box"])
                    # Ensure coords is a list
                    if isinstance(coords, (list, tuple)):
                        if len(coords) == 2:
                            # Point format [x, y] in pixels - normalize to 0-1
                            normalized_coords = [
                                coords[0] / TEST_IMAGE_WIDTH,
                                coords[1] / TEST_IMAGE_HEIGHT,
                            ]
                            logger.debug(
                                f"[extract_coord] Normalized point from {coords} to {normalized_coords}"
                            )
                            return normalized_coords, True
                        elif len(coords) == 4:
                            # Box format [x1, y1, x2, y2] in pixels - normalize to 0-1
                            normalized_coords = [
                                coords[0] / TEST_IMAGE_WIDTH,
                                coords[1] / TEST_IMAGE_HEIGHT,
                                coords[2] / TEST_IMAGE_WIDTH,
                                coords[3] / TEST_IMAGE_HEIGHT,
                            ]
                            logger.debug(
                                f"[extract_coord] Normalized box from {coords} to {normalized_coords}"
                            )
                            return normalized_coords, True
                    else:
                        logger.debug(
                            f"[extract_coord] Unexpected coord format: {coords}"
                        )
                except Exception as e:
                    logger.debug(f"[extract_coord] Error parsing coordinates: {e}")

    except Exception as e:
        logger.debug(f"[extract_coord] Error: {e}")
    return [], False


def gui_format_score(predict_str: str) -> float:
    """Calculate format score for GUI prediction."""
    try:
        parsed_actions = parse_action_to_structure_output(
            predict_str,
            IMAGE_FACTOR,
            TEST_IMAGE_HEIGHT,
            TEST_IMAGE_WIDTH,
            model_type="default",
        )
        return 1.0 if parsed_actions else 0.0
    except Exception:
        return 0.0


def gui_accuracy_score(
    predict_str: str, gt_action: str, gt_bbox: list, gt_input_text: str
) -> float:
    """Calculate accuracy score for GUI prediction (0.5 for action type, 0.5 for parameters)."""
    try:
        gt_action = gt_action.lower() if gt_action else ""

        pred_action = extract_action(predict_str).lower()
        pred_coord, has_coord = extract_coord(predict_str)
        pred_input_text = extract_input_text(predict_str)

        logger.debug(
            f"[gui_accuracy_score] gt_action: {gt_action}, pred_action: {pred_action}"
        )
        logger.debug(
            f"[gui_accuracy_score] gt_bbox: {gt_bbox}, pred_coord: {pred_coord}, has_coord: {has_coord}"
        )
        logger.debug(
            f"[gui_accuracy_score] gt_input_text: {gt_input_text}, pred_input_text: {pred_input_text}"
        )

        # Map all click variants to 'click' for the 3-action space
        action_mapping = {
            "left_single": "click",
            "left_double": "click",
            "right_single": "click",
            "click": "click",
            "type": "type",
            "scroll": "scroll",
        }

        # Normalize actions
        pred_action_normalized = action_mapping.get(pred_action, pred_action)
        gt_action_normalized = action_mapping.get(gt_action, gt_action)

        score = 0.0

        # Score calculation: 0.5 for action type, 0.5 for parameters (bbox/text)

        # 1. Action type matching (0.5 points)
        if pred_action_normalized == gt_action_normalized:
            score += 0.5
            logger.debug("[gui_accuracy_score] Action matched: +0.5 points")
        else:
            logger.debug(
                f"[gui_accuracy_score] Action mismatch: {pred_action_normalized} vs {gt_action_normalized}"
            )

        # 2. Parameter matching (0.5 points) - depends on action type
        if gt_action_normalized == "click":
            # For click: check bbox coordinates
            if gt_bbox and len(gt_bbox) > 0:
                if has_coord:
                    # Both have coordinates, calculate distance
                    # Handle different gt_bbox formats (already normalized 0-1)
                    if len(gt_bbox) == 2:
                        gt_x, gt_y = gt_bbox
                    elif len(gt_bbox) == 4:
                        gt_x = (gt_bbox[0] + gt_bbox[2]) / 2
                        gt_y = (gt_bbox[1] + gt_bbox[3]) / 2
                    else:
                        logger.debug(
                            f"[gui_accuracy_score] Invalid gt_bbox format: {gt_bbox}"
                        )
                        return score

                    # Get predicted center (already normalized 0-1)
                    if len(pred_coord) == 2:
                        pred_x, pred_y = pred_coord
                    elif len(pred_coord) == 4:
                        pred_x = (pred_coord[0] + pred_coord[2]) / 2
                        pred_y = (pred_coord[1] + pred_coord[3]) / 2
                    else:
                        logger.debug(
                            f"[gui_accuracy_score] Invalid pred_coord format: {pred_coord}"
                        )
                        return score

                    # Calculate distance in normalized space
                    distance = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5

                    # Threshold in normalized space (5% of diagonal)
                    threshold = 0.05 * (2**0.5)  # sqrt(1^2 + 1^2) ≈ 0.07

                    if distance < threshold:
                        score += 0.5
                        logger.debug(
                            f"[gui_accuracy_score] Bbox matched (distance={distance:.4f}): +0.5 points"
                        )
                    else:
                        logger.debug(
                            f"[gui_accuracy_score] Bbox too far (distance={distance:.4f}, threshold={threshold:.4f})"
                        )
                else:
                    logger.debug(
                        "[gui_accuracy_score] No predicted coordinates for click action"
                    )
            else:
                # No gt_bbox required, any click gets parameter points
                score += 0.5
                logger.debug("[gui_accuracy_score] No gt_bbox required: +0.5 points")

        elif gt_action_normalized == "type":
            # For type: check text content
            if gt_input_text and gt_input_text != "no input text":
                f1, _, _ = f1_score(pred_input_text, gt_input_text)
                if f1 >= 0.5:
                    score += 0.5
                    logger.debug(
                        f"[gui_accuracy_score] Type text matched (f1={f1:.2f}): +0.5 points"
                    )
                else:
                    logger.debug(
                        f"[gui_accuracy_score] Type text mismatch (f1={f1:.2f})"
                    )
            else:
                # No text required, any type action gets parameter points
                score += 0.5
                logger.debug("[gui_accuracy_score] No text required: +0.5 points")

        elif gt_action_normalized == "scroll":
            # For scroll: only check direction (no bbox needed)
            if gt_input_text and gt_input_text != "no input text":
                if pred_input_text.lower() == gt_input_text.lower():
                    score += 0.5
                    logger.debug(
                        "[gui_accuracy_score] Scroll direction matched: +0.5 points"
                    )
                else:
                    logger.debug(
                        f"[gui_accuracy_score] Scroll direction mismatch: {pred_input_text} vs {gt_input_text}"
                    )
            else:
                # No direction specified, any scroll gets parameter points
                score += 0.5
                logger.debug(
                    "[gui_accuracy_score] No scroll direction required: +0.5 points"
                )

        logger.debug(f"[gui_accuracy_score] Final score: {score}")
        return score

    except Exception as e:
        logger.debug(f"Error in gui_accuracy_score: {e}")
        logger.debug(f"predict_str: {predict_str}")
        logger.debug(
            f"gt_action: {gt_action}, gt_bbox: {gt_bbox}, gt_input_text: {gt_input_text}"
        )
        return 0.0


@reward(name="gui_reward")
def gui_reward(
    prediction: str,
    trajectory: List[Dict] = None,
    gt_action: str = "",
    gt_bbox: list = None,
    gt_input_text: str = "",
    **kwargs,
) -> Dict[str, float]:
    """
    Calculate GUI reward based on prediction accuracy.

    Args:
        prediction: Model prediction string
        trajectory: Conversation trajectory (optional)
        **kwargs: Additional parameters including ground truth

    Returns:
        Dictionary with reward scores
    """
    logger.debug(
        f"[gui_reward] Called with prediction: {prediction[:200] if prediction else 'None'}"
    )
    logger.debug(f"[gui_reward] kwargs keys: {list(kwargs.keys())}")

    # Handle empty predictions
    if not prediction or prediction.strip() == "":
        logger.debug("[gui_reward] Warning: Empty prediction received")
        # Check if there's a default action in trajectory
        if trajectory and len(trajectory) > 0:
            for msg in reversed(trajectory):
                if msg.get("role") == "assistant" and msg.get("content"):
                    prediction = msg["content"]
                    logger.debug(
                        f"[gui_reward] Using trajectory content as prediction: {prediction[:100]}"
                    )
                    break

        # if not prediction or prediction.strip() == "":
        #     prediction = "Thought: No response generated.\nAction: wait()"
        #     print(f"[gui_reward] Using default prediction")

    # Handle None values for parameters
    if gt_bbox is None:
        gt_bbox = []

    # Convert numpy array to list if needed
    if hasattr(gt_bbox, "tolist"):
        gt_bbox = gt_bbox.tolist()

    logger.debug(
        f"[gui_reward] gt_action: {gt_action}, gt_bbox: {gt_bbox}, gt_input_text: {gt_input_text}"
    )

    # Handle "no input text" as empty
    if gt_input_text == "no input text":
        gt_input_text = ""

    # Keep bbox in normalized coordinates (0-1 range)
    # Both prediction and ground truth use normalized coordinates

    if not gt_action and not gt_bbox and not gt_input_text:
        logger.debug(
            "[gui_reward] Warning: No ground truth data provided - returning 0 reward"
        )
        return {
            "reward": 0.0,
            "format": gui_format_score(prediction),
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    # Calculate scores
    format_score = gui_format_score(prediction)
    accuracy_score = gui_accuracy_score(prediction, gt_action, gt_bbox, gt_input_text)

    logger.debug(
        f"[gui_reward] format_score: {format_score}, accuracy_score: {accuracy_score}"
    )

    # For f1_score, create answer string for backward compatibility
    answer_dict = {"action": gt_action, "gt_bbox": gt_bbox, "input_text": gt_input_text}
    answer = json.dumps(answer_dict)
    f1, precision, recall = f1_score(prediction, answer)

    # Calculate final reward (weighted combination)
    final_reward = 0.8 * accuracy_score + 0.2 * format_score

    return {
        "reward": final_reward,
        "format": format_score,
        "accuracy": accuracy_score,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
