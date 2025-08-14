# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
from typing import List, Any, Tuple, Dict, Optional
from ..agent_base import BaseAgent
from agents.utils.ui_action_parser import parse_action_to_structure_output, IMAGE_FACTOR

# Default image dimensions
TEST_IMAGE_HEIGHT = 1080
TEST_IMAGE_WIDTH = 1920

# GUI Agent system prompt
GUI_AGENT_SYSTEM_PROMPT = """You are a GUI automation agent. Given a task and screenshot, you must analyze the screen and perform the next action.

## Response Format (REQUIRED)
You MUST always respond with exactly two lines:
Thought: [Describe what you see and what action to take]
Action: [Choose ONE action from the list below]

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='xxx') # Use escape characters \\', \\", and \\n in content part
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() # Sleep for 5s and take a screenshot to check for any changes
finished(content='xxx') # Task completed
call_user() # Submit the task and call the user when the task is unsolvable

## Examples
Example 1:
Thought: I need to click on the search button at coordinates (100, 200).
Action: click(start_box='<|box_start|>(100,200)<|box_end|>')

Example 2:
Thought: I need to type "hello world" in the text field.
Action: type(content='hello world')

## Note
- Use English in `Thought` and `Action` part
- Always provide both Thought and Action lines
- Coordinates should be in pixel values"""


class GUIAgent(BaseAgent):
    """GUI Agent for interacting with graphical user interfaces."""
    
    def __init__(
        self, 
        model_name_or_path: str, 
        template: str, 
        tools: List = None, 
        **kwargs
    ):
        """
        Initialize GUI Agent.
        
        Args:
            model_name_or_path: Path to the vision-language model
            template: Template name for formatting prompts
            tools: List of tools available to the agent
            **kwargs: Additional arguments
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            template=template,
            system_prompt=GUI_AGENT_SYSTEM_PROMPT,
            tools=tools,
            max_length=kwargs.get("max_length", 8192),
            **kwargs
        )
        self.action_counter = 0  # Track number of actions taken
        self.max_retries = 3  # Maximum retries for empty responses
        
    # def _init_llm_engine(self, model_name_or_path: str, backend: str = "vllm"):
    #     """
    #     Override to handle vision-language models properly.
        
    #     For GUI agents using vision-language models like Qwen2.5-VL,
    #     we need special handling since they're not standard causal LM models.
    #     """
    #     # For unit tests or when model loading should be skipped
    #     # if model_name_or_path == "ByteDance-Seed/UI-TARS-1.5-7B":
    #     #     # Return mock objects for testing
    #     #     print(f"[GUIAgent] Skipping actual model load for testing: {model_name_or_path}")
    #     #     return None, None, None
            
    #     # Otherwise use parent's initialization
    #     return super()._init_llm_engine(model_name_or_path, backend)
        
    def parse(self, responses: List[str], tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Parse model responses into structured messages.
        
        Args:
            responses: List of model response strings
            tools: List of available tools
            
        Returns:
            List of structured messages with tool calls
        """
        print(f"[GUIAgent.parse] Number of responses: {len(responses)}")
        print(f"[GUIAgent.parse] Raw responses type: {type(responses)}")
        
        new_messages_list = []
        
        # Process each response
        processed_responses = []
        for resp in responses:
            if resp and "Thought:" in resp and "Action:" in resp:
                processed_responses.append(resp)
            elif resp and resp.strip():
                # Try to reformat responses that don't have the expected format
                resp_lower = resp.lower()
                print(f"[GUIAgent.parse] Response missing format, reformatting: {resp[:100]}")
                
                # Check if it contains action-like content
                if any(action in resp_lower for action in ['click', 'type', 'scroll', 'drag', 'wait', 'finished']):
                    formatted_resp = f"Thought: Executing action based on response.\nAction: {resp.strip()}"
                else:
                    formatted_resp = f"Thought: {resp.strip()}\nAction: wait()"
                processed_responses.append(formatted_resp)
            else:
                # Handle empty responses with progressive default actions
                self.action_counter += 1
                if self.action_counter > 5:
                    processed_responses.append("Thought: Multiple attempts made. Completing task.\nAction: finished(content='Task attempted')")
                else:
                    processed_responses.append(f"Thought: Processing the screen (attempt {self.action_counter}).\nAction: wait()")
        
        responses = processed_responses
        
        # Log responses for debugging
        for idx, resp in enumerate(responses[:3]):  # Log first 3 responses
            if resp:
                print(f"[GUIAgent.parse] Response {idx} length: {len(resp)}, preview: {resp[:200]}")
            else:
                print(f"[GUIAgent.parse] Response {idx} is None or empty")
        
        # Parse actions from responses
        action_list = []
        for response in responses:
            parsed = parse_action_to_structure_output(
                response, 
                IMAGE_FACTOR, 
                TEST_IMAGE_HEIGHT, 
                TEST_IMAGE_WIDTH
            )
            action_list.append(parsed)
        
        # Create messages with tool calls
        for i, (response, actions) in enumerate(zip(responses, action_list)):
            print(f"[GUIAgent.parse] Processing response {i+1}: response_length={len(response) if response else 0}, actions={actions}")
            
            tool_calls = []
            
            if actions is not None and len(actions) > 0:
                if len(actions) > 1:
                    print(f"[GUIAgent.parse] Warning: Multiple actions found ({len(actions)}), using first one")
                action = actions[0]
                tool_calls = [{
                    "id": str(i),
                    "type": "function",
                    "function": {
                        "name": "pyautogui_code_generator",
                        "arguments": json.dumps({"action": action})
                    }
                }]
            else:
                # If no action was parsed, create a default wait action
                print(f"[GUIAgent.parse] No action parsed from response, creating default wait action")
                default_action = {
                    "action_type": "wait",
                    "action_inputs": {},
                    "thought": "Waiting to analyze the screen",
                    "reflection": None
                }
                tool_calls = [{
                    "id": str(i),
                    "type": "function",
                    "function": {
                        "name": "pyautogui_code_generator",
                        "arguments": json.dumps({"action": default_action})
                    }
                }]
            
            # Determine status - terminal if action is "finished" or "call_user"
            status = "continue"  # Default to continue
            if actions and isinstance(actions[0], dict):
                action_type = actions[0].get("action_type", "")
                print(f"[GUIAgent.parse] Action type: {action_type}")
                if action_type in ["finished", "call_user"]:
                    status = "terminal"
            
            message = {
                "role": "assistant",
                "content": [{"type": "text", "text": response}] if response else [{"type": "text", "text": ""}],
                "tool_calls": tool_calls,
                "loss": True,
                "status": status
            }
            print(f"[GUIAgent.parse] Created message with status={status}, tool_calls={len(tool_calls)}, content_length={len(response)}")
            new_messages_list.append(message)
        
        return new_messages_list
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format messages for the vision-language model.
        
        Args:
            messages: List of messages to format
            
        Returns:
            Formatted messages suitable for VLM input
        """
        formatted_messages = []
        
        for msg in messages:
            formatted_msg = {
                "role": msg.get("role"),
                "content": msg.get("content", "")
            }
            
            # Handle image content if present
            if "images" in msg:
                # Convert images to appropriate format for the model
                formatted_msg["images"] = msg["images"]
            
            formatted_messages.append(formatted_msg)
        
        return formatted_messages