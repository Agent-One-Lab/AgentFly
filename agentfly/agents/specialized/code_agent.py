import json
from typing import Any, Callable, Dict, List, Tuple
try:
    from verl.protocol import DataProto
except ImportError:
    pass
from ..agent_base import BaseAgent
import torch
import numpy as np
import re


def extract_python_code_markdown(text):
    """
    Extracts all Python code blocks from a given markdown-formatted string.
    
    A Python code block is defined as text that starts with ```python 
    and ends with ```.

    Args:
        text (str): The input string potentially containing multiple Python code blocks.

    Returns:
        List[str]: A list of Python code blocks (with markdown syntax removed).
    """
    # The regex matches text starting with ```python, capturing everything until the closing ```
    pattern = r"```python\s*(.*?)\s*```"
    # Using re.DOTALL to allow the dot to match newlines
    code_blocks = re.findall(pattern, text, re.DOTALL)
    return code_blocks


CodeAgentSystemPrompt = """The user asks a question, and you solve it in a multi-turn manner. During each turn, you should give the python code. The code interpreter will give you the output of executing the code. You should repeat the process until you find the final answer.

The code is in markdown format, enclosed with ```python and ```. In each turn, only use code to solve the question for one step, and you can stop to wait for the execution result. Do not try to solve the whole question with one code snippet. Only when you get the final answer, can you give the answer enclosed with <answer> and </answer>. 
"""

# CodeAgentSystemPrompt = r"""The user asks a question, and you solve it. You first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. And your final answer will be extracted automatically by the \boxed{} tag."""

class CodeAgent(BaseAgent):
    def __init__(
            self,
            model_name_or_path: str,
            template: str,
            tools: List = None,
            **kwargs
        ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            template=template,
            system_prompt=CodeAgentSystemPrompt,
            tools=tools,
            **kwargs
        )


    def parse(self, responses: List[str], tools: List[Any]) -> Tuple[dict, int, int]:
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
        codes_list = [extract_python_code_markdown(response) for response in responses]

        new_messages_list = []
        for response, codes in zip(responses, codes_list):
            if len(codes) != 1:
                # 1. No code block 2. Multiple code blocks
                # We set multiple code blocks as failed because it does not follow our expected format.
                # In this case, the interaction will be terminated and the reward function will check the reward.
                tool_calls = []
            else:
                code = codes[0]
                tool_calls = [{
                    "id": None,
                    "type": "function",
                    "function": {
                        "name": "code_interpreter",
                        "arguments": json.dumps({"code": code})
                    }
                }]
            message = {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
                "tool_calls": tool_calls,
                "loss": True,
                "status": "continue" if len(tool_calls) == 1 else "terminal"
            }
            new_messages_list.append(message)

        return new_messages_list


if __name__ == "__main__":
    """python -m agents.agents.templates.template"""
    pass