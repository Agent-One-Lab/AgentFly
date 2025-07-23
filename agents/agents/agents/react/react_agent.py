

import json
from typing import Any, Dict, List, Optional
from ..utils.json import jsonish
from ...tools.tool_base import Tool
try:
    from verl.protocol import DataProto
except ImportError:
    pass
from ..agent_base import BaseAgent
import torch
import numpy as np
import re

def parse_react_step(text: str) -> Dict[str, Optional[str]]:
    """
    Parse a single ReAct-style step (one Thought→Action→Input) into its components.

    Args:
        text: A string containing exactly one Thought:, one Action:, and one Input:.

    Returns:
        A dict with keys 'thought', 'action', and 'input', or None if not found.
    """
    pattern = re.compile(
        r"Thought:\s*(?P<thought>.*?)\s*"
        r"Action:\s*(?P<action>.*?)\s*"
        r"Input:\s*(?P<input>.*)",
        re.IGNORECASE | re.DOTALL
    )
    m = pattern.search(text)
    if not m:
        return {"thought": None, "action": None, "input": None}

    return {
        "thought": m.group("thought").strip(),
        "action": m.group("action").strip(),
        "input": m.group("input").strip(),
    }

def extract_tool_calls(action_input: str) -> List[Dict]:
    if action_input is None:
        return []
    
    tool_call_str = ""
    # Extract the tool call from the action input
    # 1. Extract with qwen style
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    m = pattern.search(action_input)
    # If we find a tool call, extract it
    if m:
        tool_call_str = m.group(1).strip()
        try:
            tool_call = jsonish(tool_call_str)
            return [tool_call]
        except:
            pass
    
    # 2. Extract directly
    try:
        tool_call = jsonish(action_input)
        return [tool_call]
    except:
        pass
    
    return []


ReactSystemPromptTemplate = """You are a ReAct-style agent. When you receive a user query, in each step, you must:

1. **Think** in natural language about what to do next.  
   - Prefix each internal reasoning step with `Thought:`.  
2. **Act** by calling one of your available tools. The tools must be selected from the given list.
   - Prefix with `Action:` and the name of the tool.
3. **Input** the tool's input. The input must be a valid JSON object.
   - Prefix with `Input:` and the input to the tool.
4. Observe the tool's output.

You must repeat Think→Act→Observe until you're ready to give a final answer.  
When finished, output one final line prefixed `Answer:` with your concise solution.

{task_info}{tools}"""

TaskInfoTemplate = """**Task Information**
{task_info}
"""

ToolSchemasTemplate = """**Available Tools**
{tool_schemas}
"""

"""**Example Thought-Action-Input**
Thought: I need to find the weather in San Francisco today.
Action: search
Input: {{"query": "weather in San Francisco today"}}"""


class ReactAgent(BaseAgent):
    def __init__(self,
            model_name_or_path: str,
            tools: List[Tool],
            task_info: str = None,
            **kwargs
        ):
        schema_list = [tool.schema for tool in tools]
        if task_info is None or task_info == "":
            task_info = ""
        else:
            task_info = TaskInfoTemplate.format(task_info=task_info)
        
        tool_schemas = ToolSchemasTemplate.format(tool_schemas="\n".join(json.dumps(schema, indent=4) for schema in schema_list))
        system_prompt = ReactSystemPromptTemplate.format(task_info=task_info, tools=tool_schemas)

        super().__init__(
            model_name_or_path=model_name_or_path,
            tools=tools,
            system_prompt=system_prompt,
            max_length=8192,
            **kwargs
        )
        
    def parse(self, responses: List[str], tools: List[Any]) -> List[Dict]:
        """
        Generates an assistant message compatible with tool-calling.
        Returns:
            List of messages with the following format:
            message: A dict with keys "role", "content", and "tool_calls".
                tool_calls: A list of tool calls with the following format:
                    {
                        "id": None,
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": ""
                        }
                    }
        """
        # print(f"responses: {responses}")
        thought_actions = [parse_react_step(response) for response in responses]

        new_messages_list = []
        for response, thought_action in zip(responses, thought_actions):
            
            thought = thought_action["thought"]
            action = thought_action["action"]
            action_input = thought_action["input"]
            if action is None:
                tool_calls = []
            else:
                tool_calls = extract_tool_calls(action_input)
            
            formatted_tool_calls = []
            # We only support one tool call for now
            if len(tool_calls) == 1:
                tool_call = tool_calls[0]
                try:
                    tool_call = json.loads(tool_call)
                    # {"name": "...", "arguments": "..."}
                    if "name" in tool_call and "arguments" in tool_call:
                        name = tool_call["name"]
                        arguments = tool_call["arguments"]
                    # {"param1": "...", "param2": "..."}
                    else:
                        name = action
                        arguments = tool_call
                    formatted_tool_calls.append({
                        "id": None,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments
                        }
                    })
                except Exception as e:
                    name = action
                    arguments = tool_call
            else:
                pass

            message = {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
                "tool_calls": formatted_tool_calls,
                "loss": True
            }
            new_messages_list.append(message)

        return new_messages_list
    

if __name__ == "__main__":
    pass