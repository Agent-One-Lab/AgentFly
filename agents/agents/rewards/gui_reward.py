# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import re
import json
import ast
from typing import Dict, Any, List, Tuple, Optional

from .reward_base import reward
from agents.utils.ui_action_parser import parse_action_to_structure_output, IMAGE_FACTOR

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
    
    precision = len(common_tokens) / len(normalized_prediction) if normalized_prediction else 0.0
    recall = len(common_tokens) / len(normalized_ground_truth) if normalized_ground_truth else 0.0
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall


def extract_action(content: str) -> str:
    """Extract action type from response content."""
    try:
        parsed = parse_action_to_structure_output(content, IMAGE_FACTOR, TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, model_type="default")
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
        print(f"[extract_action] Error: {e}")
    return "no action"


def extract_input_text(content: str) -> str:
    """Extract input text from action content."""
    try:
        parsed = parse_action_to_structure_output(content, IMAGE_FACTOR, TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, model_type="default")
        if parsed and len(parsed) > 0:
            action_dict = parsed[0]
            action_type = action_dict.get("action_type")
            action_inputs = action_dict.get("action_inputs", {})
            
            # Extract text based on action type
            if action_type == 'type':
                return action_inputs.get('content', '')
            elif action_type == 'scroll':
                return action_inputs.get('direction', 'down')
            elif action_type == 'hotkey':
                return action_inputs.get('key', action_inputs.get('hotkey', ''))
            elif action_type == 'finished':
                return action_inputs.get('content', '')
            
            return ""
    except Exception as e:
        print(f"[extract_input_text] Error: {e}")
    return ""


def extract_coord(content: str) -> Tuple[list, bool]:
    """Extract coordinates from action content."""
    try:
        parsed = parse_action_to_structure_output(content, IMAGE_FACTOR, TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, model_type="default")
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
                            # Point format [x, y]
                            return list(coords), True
                        elif len(coords) == 4:
                            # Box format [x1, y1, x2, y2]
                            return list(coords), True
                    else:
                        print(f"[extract_coord] Unexpected coord format: {coords}")
                except Exception as e:
                    print(f"[extract_coord] Error parsing coordinates: {e}")
                
    except Exception as e:
        print(f"[extract_coord] Error: {e}")
    return [], False


def gui_format_score(predict_str: str) -> float:
    """Calculate format score for GUI prediction."""
    try:
        parsed_actions = parse_action_to_structure_output(predict_str, IMAGE_FACTOR, TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, model_type="default")
        return 1.0 if parsed_actions else 0.0
    except Exception:
        return 0.0


def gui_accuracy_score(predict_str: str, gt_action: str, gt_bbox: list, gt_input_text: str) -> float:
    """Calculate accuracy score for GUI prediction."""
    try:
        gt_action = gt_action.lower() if gt_action else ''

        pred_action = extract_action(predict_str).lower()
        pred_coord, has_coord = extract_coord(predict_str)
        pred_input_text = extract_input_text(predict_str)
        
        print(f"[gui_accuracy_score] gt_action: {gt_action}, pred_action: {pred_action}")
        print(f"[gui_accuracy_score] gt_bbox: {gt_bbox}, pred_coord: {pred_coord}, has_coord: {has_coord}")
        
        # Normalize action types for comparison
        action_mapping = {
            'left_single': 'click',
            'left_double': 'left_double',
            'right_single': 'right_single',
            'select': 'drag',
            'press': 'hotkey',
            'keydown': 'hotkey',
            'release': 'hotkey',
            'keyup': 'hotkey'
        }
        
        # Normalize predicted action
        pred_action_normalized = action_mapping.get(pred_action, pred_action)
        gt_action_normalized = action_mapping.get(gt_action, gt_action)
        
        # For click-related actions, treat them as the same category
        click_actions = ['click', 'left_single', 'left_double', 'right_single']
        if pred_action in click_actions and gt_action in click_actions:
            # Actions are in the same category, continue with coordinate check
            pass
        elif pred_action_normalized != gt_action_normalized:
            return 0.0

        # Check accuracy for different action types
        if gt_action in ["click", "left_single", "left_double", "right_single"]:
            if has_coord and gt_bbox:
                # Handle different gt_bbox formats
                if len(gt_bbox) == 2:
                    # Point format
                    gt_x, gt_y = gt_bbox
                elif len(gt_bbox) == 4:
                    # Box format - use center
                    gt_x = (gt_bbox[0] + gt_bbox[2]) / 2
                    gt_y = (gt_bbox[1] + gt_bbox[3]) / 2
                else:
                    return 0.0
                
                # Get predicted center based on format
                if len(pred_coord) == 2:
                    # Point format
                    pred_x, pred_y = pred_coord
                elif len(pred_coord) == 4:
                    # Box format - use center
                    pred_x = (pred_coord[0] + pred_coord[2]) / 2
                    pred_y = (pred_coord[1] + pred_coord[3]) / 2
                else:
                    return 0.0
                
                # Calculate distance
                distance = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
                
                # Use a threshold based on screen size (e.g., 5% of screen diagonal)
                screen_diagonal = (TEST_IMAGE_WIDTH ** 2 + TEST_IMAGE_HEIGHT ** 2) ** 0.5
                threshold = 0.05 * screen_diagonal
                
                return 1.0 if distance < threshold else 0.0
            return 0.0 if gt_bbox else 1.0  # If no bbox in gt, any click is correct

        elif gt_action in ['type', 'scroll']:
            f1, _, _ = f1_score(pred_input_text, gt_input_text)
            return 1.0 if f1 >= 0.5 else 0.0
        
        elif gt_action == 'drag':
            # For drag, check start position
            if has_coord and gt_bbox:
                if len(gt_bbox) >= 4:
                    # Compare start position
                    if len(pred_coord) == 2:
                        pred_x, pred_y = pred_coord
                    elif len(pred_coord) >= 4:
                        pred_x = (pred_coord[0] + pred_coord[2]) / 2
                        pred_y = (pred_coord[1] + pred_coord[3]) / 2
                    else:
                        return 0.0
                    
                    if len(gt_bbox) == 4:
                        gt_start_x, gt_start_y = gt_bbox[0], gt_bbox[1]
                    else:
                        gt_start_x = (gt_bbox[0] + gt_bbox[2]) / 2
                        gt_start_y = (gt_bbox[1] + gt_bbox[3]) / 2
                    
                    distance = ((pred_x - gt_start_x) ** 2 + (pred_y - gt_start_y) ** 2) ** 0.5
                    screen_diagonal = (TEST_IMAGE_WIDTH ** 2 + TEST_IMAGE_HEIGHT ** 2) ** 0.5
                    threshold = 0.05 * screen_diagonal
                    
                    return 1.0 if distance < threshold else 0.0
            return 0.0
        
        elif gt_action == 'hotkey':
            # For hotkey, compare the key combinations
            return 1.0 if pred_input_text.lower() == gt_input_text.lower() else 0.0
        
        elif gt_action in ['wait', 'finished', 'call_user']:
            # These actions don't require parameters
            return 1.0
        
        return 0.0
    except Exception as e:
        print(f"Error in gui_accuracy_score: {e}")
        print(f"predict_str: {predict_str}")
        print(f"gt_action: {gt_action}, gt_bbox: {gt_bbox}, gt_input_text: {gt_input_text}")
        return 0.0


@reward(name="gui_reward")
def gui_reward(prediction: str, trajectory: List[Dict] = None, gt_action: str = "", gt_bbox: list = None, gt_input_text: str = "", **kwargs) -> Dict[str, float]:
    """
    Calculate GUI reward based on prediction accuracy.
    
    Args:
        prediction: Model prediction string
        trajectory: Conversation trajectory (optional)
        **kwargs: Additional parameters including ground truth
        
    Returns:
        Dictionary with reward scores
    """
    print(f"[gui_reward] Called with prediction: {prediction[:200] if prediction else 'None'}")
    print(f"[gui_reward] kwargs keys: {list(kwargs.keys())}")
    
    # Handle empty predictions
    if not prediction or prediction.strip() == "":
        print(f"[gui_reward] Warning: Empty prediction received")
        # Check if there's a default action in trajectory
        if trajectory and len(trajectory) > 0:
            for msg in reversed(trajectory):
                if msg.get('role') == 'assistant' and msg.get('content'):
                    prediction = msg['content']
                    print(f"[gui_reward] Using trajectory content as prediction: {prediction[:100]}")
                    break
        
        # if not prediction or prediction.strip() == "":
        #     prediction = "Thought: No response generated.\nAction: wait()"
        #     print(f"[gui_reward] Using default prediction")
    
    # Handle None values for parameters
    if gt_bbox is None:
        gt_bbox = []
    
    # Convert numpy array to list if needed
    if hasattr(gt_bbox, 'tolist'):
        gt_bbox = gt_bbox.tolist()
    
    print(f"[gui_reward] gt_action: {gt_action}, gt_bbox: {gt_bbox}, gt_input_text: {gt_input_text}")
    
    # Handle "no input text" as empty
    if gt_input_text == "no input text":
        gt_input_text = ""
        
    # Convert normalized bbox to pixel coordinates if needed
    if gt_bbox and len(gt_bbox) > 0 and all(0 <= v <= 1 for v in gt_bbox):
        print(f"[gui_reward] Converting normalized bbox to pixel coordinates")
        if len(gt_bbox) == 2:
            gt_bbox = [gt_bbox[0] * TEST_IMAGE_WIDTH, gt_bbox[1] * TEST_IMAGE_HEIGHT]
        elif len(gt_bbox) == 4:
            gt_bbox = [
                gt_bbox[0] * TEST_IMAGE_WIDTH,
                gt_bbox[1] * TEST_IMAGE_HEIGHT,
                gt_bbox[2] * TEST_IMAGE_WIDTH,
                gt_bbox[3] * TEST_IMAGE_HEIGHT
            ]
        print(f"[gui_reward] Converted bbox: {gt_bbox}")
    
    if not gt_action and not gt_bbox and not gt_input_text:
        print(f"[gui_reward] Warning: No ground truth data provided - returning 0 reward")
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
    
    print(f"[gui_reward] format_score: {format_score}, accuracy_score: {accuracy_score}")
    
    # For f1_score, create answer string for backward compatibility
    answer_dict = {
        "action": gt_action,
        "gt_bbox": gt_bbox,
        "input_text": gt_input_text
    }
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