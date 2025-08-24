# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import re
import ast
import math
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 1  # Changed to match gui_reward.py
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def convert_point_to_coordinates(text: str, is_answer: bool = False) -> str:
    """Convert point format to coordinates."""
    pattern = r"<point>(\d+)\s+(\d+)</point>"

    def replace_match(match):
        x1, y1 = map(int, match.groups())
        x = (x1 + x1) // 2  # Truncate to integer
        y = (y1 + y1) // 2  # Truncate to integer
        if is_answer:
            return f"({x},{y})"  # Only return (x, y) format
        return f"({x},{y})"  # Return the format with tags

    # Remove [EOS] and replace <bbox> coordinates
    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()


def parse_action(action_str: str) -> Optional[Dict[str, Any]]:
    """Parse an action string into function name and arguments."""
    try:
        # Parse the string to AST node
        node = ast.parse(action_str, mode='eval')

        # Ensure the node is an expression
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # Get the body of the expression
        call = node.body

        # Ensure the body is a function call
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # Get the function name
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # Get the keyword arguments
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # Handle different types of values
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # Compatible with old version Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {'function': func_name, 'args': kwargs}

    except Exception as e:
        logger.debug(f"Failed to parse action '{action_str}': {e}")
        return None


def escape_single_quotes(text: str) -> str:
    """Escape unescaped single quotes."""
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def linear_resize(height: int,
                  width: int,
                  factor: int = IMAGE_FACTOR,
                  min_pixels: int = MIN_PIXELS,
                  max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
    """Resize image to fit within pixel limits while maintaining aspect ratio."""
    if width * height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(height * resize_factor)
    return height, width


def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
    """
    Rescales the image so that:
    1. Both dimensions are divisible by 'factor'.
    2. Total pixels is within [min_pixels, max_pixels].
    3. Aspect ratio is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def parse_action_to_structure_output(text: str,
                                     factor: int,
                                     origin_resized_height: int,
                                     origin_resized_width: int,
                                     model_type: str = "qwen25vl",
                                     max_pixels: int = 16384 * 28 * 28,
                                     min_pixels: int = 100 * 28 * 28) -> Optional[List[Dict[str, Any]]]:
    """Parse action text to structured output."""
    logger.debug(f"[parse_action_to_structure_output] Input text: {text[:500] if text else 'Empty text'}")
    
    # Handle empty or None responses
    if not text:
        logger.debug(f"[parse_action_to_structure_output] Empty text, returning None")
        return None
        
    text = text.strip()

    # Handle various point/box formats
    if "<point>" in text:
        text = convert_point_to_coordinates(text)
    if "start_point=" in text:
        text = text.replace("start_point=", "start_box=")
    if "end_point=" in text:
        text = text.replace("end_point=", "end_box=")
    if "point=" in text:
        text = text.replace("point=", "start_box=")

    smart_resize_height, smart_resize_width = origin_resized_height, origin_resized_width
    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height,
            origin_resized_width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels)

    # Extract thought and reflection
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action: |$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action: |$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        thought_hint = "Thought: "
    
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    
    if "Action:" not in text:
        logger.debug(f"[parse_action_to_structure_output] No 'Action:' found in text, returning None")
        return None
        
    action_str = text.split("Action: ")[-1]
    logger.debug(f"[parse_action_to_structure_output] Extracted action string: {action_str[:200]}")

    # Parse multiple actions
    tmp_all_action = action_str.split(")\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            if not action_str.strip().endswith(")"):
                action_str = action_str.strip() + ")"
            # Handle type content escaping
            pattern = r"type\(content='(.*?)'\)"
            if re.search(pattern, action_str):
                content = re.sub(pattern, lambda m: m.group(1), action_str)
                action_str = escape_single_quotes(content)
                action_str = "type(content='" + action_str + "')"
        if not action_str.strip().endswith(")"):
            action_str = action_str.strip() + ")"
        all_action.append(action_str)

    parsed_actions = [
        parse_action(action.replace("\n", "\\n").lstrip())
        for action in all_action
    ]
    
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance is None:
            logger.debug(f"Action can't parse: {raw_str}")
            continue
            
        action_type = action_instance["function"]
        params = action_instance["args"]

        action_inputs = {}
        for param_name, param in params.items():
            if param is None or param == "": 
                continue
            
            # Only apply lstrip to string parameters
            if isinstance(param, str):
                param = param.lstrip()
            
            action_inputs[param_name.strip()] = param

            # Handle box coordinates (only for string parameters)
            if isinstance(param, str) and ("start_box" in param_name or "end_box" in param_name):
                ori_box = param
                # Remove box tags if present
                ori_box = ori_box.replace("<|box_start|>", "").replace("<|box_end|>", "")
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                try:
                    for num in numbers:
                        float(num.strip())
                except ValueError:
                    logger.debug(f"Warning: Invalid coordinate format in '{param_name}': '{ori_box}'")
                    return None

                # Convert coordinates based on model type
                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(float(num / smart_resize_height))
                        else:
                            float_numbers.append(float(num / smart_resize_width))
                else:
                    # For IMAGE_FACTOR = 1, keep coordinates as pixel values
                    float_numbers = [float(num.strip()) for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [
                        float_numbers[0], float_numbers[1], 
                        float_numbers[0], float_numbers[1]
                    ]
                action_inputs[param_name.strip()] = str(float_numbers)

        # Normalize action types for consistency
        normalized_action_type = action_type
        if action_type in ["left_single", "left_double", "right_single"]:
            normalized_action_type = "click"
        elif action_type in ["press", "keydown", "release", "keyup"]:
            normalized_action_type = "hotkey"
        elif action_type in ["select"]:
            normalized_action_type = "drag"
            
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": normalized_action_type,
            "action_inputs": action_inputs,
            "text": text,
        })
    
    return actions


def parsing_response_to_pyautogui_code(responses: List[Dict[str, Any]],
                                       image_height: int,
                                       image_width: int,
                                       input_swap: bool = True) -> str:
    """Convert parsed responses to PyAutoGUI code."""
    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    
    for response_id, response in enumerate(responses):
        observation = response.get("observation", "")
        thought = response.get("thought", "")

        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(1)\n"

        action_type = response.get("action_type")
        action_inputs = response.get("action_inputs", {})

        if action_type == "hotkey":
            hotkey = action_inputs.get("key", action_inputs.get("hotkey", ""))
            # Convert arrow keys
            hotkey = hotkey.replace("arrowleft", "left").replace("arrowright", "right")
            hotkey = hotkey.replace("arrowup", "up").replace("arrowdown", "down")
            
            if hotkey:
                keys = hotkey.split()
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"

        elif action_type == "type":
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content.rstrip("\\n").rstrip("\n")
            
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        elif action_type in ["drag", "select"]:
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n")

        elif action_type == "scroll":
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
            else:
                x = None
                y = None
            
            direction = action_inputs.get("direction", "")
            
            if x is None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"

        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code