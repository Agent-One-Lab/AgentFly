# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from ...tool_base import tool
from agents.utils.ui_action_parser import parsing_response_to_pyautogui_code

# Default image dimensions for UI interactions
DEFAULT_IMAGE_HEIGHT = 1080
DEFAULT_IMAGE_WIDTH = 1920


@tool(name="pyautogui_code_generator")
def pyautogui_code_generator(action: dict, **kwargs) -> str:
    """
    Generate PyAutoGUI code from a structured action dictionary.
    
    Args:
        action (dict): Dictionary containing action_type, action_inputs, thought, etc.
        **kwargs: Additional parameters like image dimensions
    
    Returns:
        PyAutoGUI code string or execution result
    """
    print(f"[pyautogui_code_generator] Received action: {action}")
    
    # Extract image dimensions from kwargs or use defaults
    image_height = kwargs.get("image_height", DEFAULT_IMAGE_HEIGHT)
    image_width = kwargs.get("image_width", DEFAULT_IMAGE_WIDTH)
    
    # Handle the action
    if isinstance(action, str):
        # Try to parse if it's a JSON string
        try:
            action = json.loads(action)
        except json.JSONDecodeError:
            return f"Error: Invalid action format - {action}"
    
    if not isinstance(action, dict):
        return f"Error: Action must be a dictionary, got {type(action)}"
    
    # Check if this is a terminal action
    action_type = action.get("action_type", "")
    if action_type in ["finished", "call_user"]:
        content = action.get("action_inputs", {}).get("content", "Task completed")
        return f"Task completed: {content}"
    
    # Generate PyAutoGUI code for the action
    try:
        pyautogui_code = parsing_response_to_pyautogui_code(
            [action],
            image_height=image_height,
            image_width=image_width,
            input_swap=True
        )
        
        # For non-terminal actions, return the code
        if pyautogui_code == "DONE":
            return "Task completed successfully"
        
        return f"Generated PyAutoGUI code:\n{pyautogui_code}"
        
    except Exception as e:
        return f"Error generating PyAutoGUI code: {str(e)}"


@tool(name="capture_screenshot")
def capture_screenshot(**kwargs) -> str:
    """
    Capture a screenshot of the current screen.
    
    Returns:
        Base64 encoded screenshot or error message
    """
    try:
        import pyautogui
        import base64
        from io import BytesIO
        
        # Take screenshot
        screenshot = pyautogui.screenshot()
        
        # Convert to base64
        buffered = BytesIO()
        screenshot.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"Screenshot captured successfully (base64): {img_str[:100]}..."
        
    except ImportError:
        return "Error: PyAutoGUI not installed. Please install it to capture screenshots."
    except Exception as e:
        return f"Error capturing screenshot: {str(e)}"