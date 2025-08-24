import os
import logging
from typing import Any, List, Tuple
import re
import json
from ..agent_base import BaseAgent

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.environ.get("VERL_LOGGING_LEVEL", "INFO"))

def jsonish_to_dict(text: str) -> dict:
    """
    Convert a string that *looks* like JSON but may contain
    raw newlines or bad escape sequences (e.g. \')
    into a real dict.
    """
    # 1 ⟶ escape every back‑slash that *isn't* already part of a valid JSON escape
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text.strip())
    
    # 2 ⟶ turn every real newline into the two‑character escape \n
    text = text.replace('\n', r'\n')
    
    # 3 ⟶ now it's valid JSON
    return json.loads(text)
    
def parse_thinking_response(response) -> Tuple[str, str, List[dict]]:
    """
    Parse the thinking response and return the thinking, response, and tool calls.
    
    Args:
        response: A string in the format "<think> think process </think> [<answer>] <tool_call> {"name": <tool_name>, "arguments": <args_json_object>}</tool_call> [</answer>]"
        where <answer> and </answer> tags are optional.
    
    Returns:
        Tuple[str, str, List[dict]]: (thinking, response, tool_calls)
    """
    LOGGER.debug(f"Response: {response}")
    thinking = ""
    answer = ""
    tool_calls = []
    
    # Extract thinking process
    try:
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
        else:
            thinking = ""
    except Exception as e:
        LOGGER.debug(f"Error parsing thinking: {e}")
        thinking = ""
    
    # Extract answer content (optional)
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # If no answer tags, get everything after </think>
            after_think = re.split(r'</think>', response, maxsplit=1)
            if len(after_think) > 1:
                answer = after_think[1].strip()
            else:
                answer = ""
    except Exception as e:
        LOGGER.debug(f"Error parsing answer: {e}")
        answer = ""

    
    # Extract tool calls from the entire response after </think>
    try:
        # First try to find tool calls in the answer section
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, answer, re.DOTALL)
        
        # If no tool calls found in answer, search in the entire response after </think>
        if not tool_call_matches:
            after_think = re.split(r'</think>', response, maxsplit=1)
            if len(after_think) > 1:
                remaining_text = after_think[1]
                tool_call_matches = re.findall(tool_call_pattern, remaining_text, re.DOTALL)
        
        for tool_call_str in tool_call_matches:
            try:
                tool_calls.append(tool_call_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, skip this tool call
                continue
    except Exception as e:
        LOGGER.debug(f"Error parsing tool calls: {e}")
        tool_calls = []

    valid_tool_calls = []
    for tool_call in tool_calls:
        try:
            tool_call_dict = jsonish_to_dict(tool_call)
            if "name" in tool_call_dict and "arguments" in tool_call_dict:
                valid_tool_calls.append(tool_call_dict)
        except Exception as e:
            LOGGER.debug(f"Error parsing tool call: {tool_call}")


    LOGGER.debug(f"Thinking: {thinking}")
    LOGGER.debug(f"Answer: {answer}")
    LOGGER.debug(f"Tool calls: {tool_calls}")
    
    return thinking, answer, tool_calls, valid_tool_calls


class ThinkAgent(BaseAgent):
    def __init__(
            self,
            model_name_or_path: str,
            template: str = "qwen2.5-think",
            tools: List = None,
            **kwargs
        ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            template=template,
            tools=tools,
            max_length=8192,
            **kwargs
        )


    def parse(self, responses: List[str], tools: List[Any]) -> Tuple[dict, int, int]:
        thinking_answer_tool_calls = [parse_thinking_response(response) for response in responses]

        new_messages_list = []
        for i, (thinking, answer, tool_calls, valid_tool_calls) in enumerate(thinking_answer_tool_calls):
            # We only allow one tool call for now
            # TODO: Support multiple tool calls
            if len(valid_tool_calls) > 1:
                valid_tool_calls = valid_tool_calls[:1]
            message = {
                "role": "assistant",
                "content": [{"type": "text", "text": responses[i]}],
                "tool_calls": [
                    {
                        "id": None,
                        "type": "function",
                        "function": tool_call
                    } for tool_call in valid_tool_calls
                ],
                "loss": True,
                "status": "continue" if len(valid_tool_calls) > 0 else "terminal"
            }
            new_messages_list.append(message)

        return new_messages_list
