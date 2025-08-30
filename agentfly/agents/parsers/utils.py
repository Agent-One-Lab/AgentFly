import re
from typing import List, Dict
import json
from ..utils.json import jsonish

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