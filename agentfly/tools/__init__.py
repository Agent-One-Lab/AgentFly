# Global tool registry
TOOL_REGISTRY = {}
TOOL_FACTORY = {}

from typing import List
from .tool_base import Tool, hallucination_tool, invalid_input_tool, tool, submit_tool_call, submit_tool_calls
from .src.code.tools import code_interpreter
from .src.alfworld.tools import alfworld_step, alfworld_get_task_objective, alfworld_get_admissible_commands, alfworld_reset
from .src.search.google_search import google_search_serper
from .src.search.dense_retriever import dense_retrieve
from .src.search.async_dense_retriever import asyncdense_retrieve
# from .src.search.http_retriever import http_retrieve
from .src.webshop.tools import webshop_browser
from .src.react.tools import answer_qa, answer_math
from .src.search.async_dense_retriever import asyncdense_retrieve
from .src.scienceworld.tools import scienceworld_explorer
from .src.ui.tools import pyautogui_code_generator

# Export the tools
__all__ = [
    "asyncdense_retrieve",
    "dense_retrieve"
    "http_retrieve",
    "code_interpreter",
    "alfworld_step",
    "alfworld_reset", 
    "alfworld_get_admissible_commands",
    "google_search_serper",
    "answer_qa",
    "answer_math",
    "hallucination_tool",
    "invalid_input_tool",
    "submit_tool_call",
    "submit_tool_calls",
    "tool",
    "webshop_browser"
    "alfworld_get_task_objective"
    "alfworld_reset"
    "asyncdense_retrieve"
    "pyautogui_code_generator"
    # "current_env"
]

# Add explicit tools in case they weren't auto-registered
EXPLICIT_TOOLS = {
    "asyncdense_retrieve": asyncdense_retrieve,
    "dense_retrieve": dense_retrieve,
    # "http_retrieve": http_retrieve,
    "code_interpreter": code_interpreter,
    "alfworld_step": alfworld_step,
    "alfworld_reset": alfworld_reset,
    "alfworld_get_task_objective": alfworld_get_task_objective,
    "alfworld_get_admissible_commands": alfworld_get_admissible_commands,
    "google_search": google_search_serper,
    "answer_qa": answer_qa,
    "answer_math": answer_math,
    "hallucination_tool": hallucination_tool,
    "invalid_input_tool": invalid_input_tool,
    "dense_retrieve": dense_retrieve,
    "pyautogui_code_generator": pyautogui_code_generator
}

# Update the registry with explicit tools
for _name, _tool in EXPLICIT_TOOLS.items():
    if _name not in TOOL_REGISTRY:
        TOOL_REGISTRY[_name] = _tool

def register_tool(tool_name, tool_func):
    """
    Register a tool in the tool registry.
    
    Args:
        tool_name: The name of the tool
        tool_func: The tool function or BaseTool instance
    """
    TOOL_REGISTRY[tool_name] = tool_func

def get_tool_from_name(tool_name: str) -> Tool:
    """
    Get a tool instance from its name.
    """
    return TOOL_REGISTRY[tool_name]

def get_tools_from_names(tool_names: List[str]) -> List[Tool]:
    """
    Get tool instances from their names.
    
    Args:
        tool_names: List of tool names
        
    Returns:
        List of BaseTool instances
        
    Raises:
        KeyError: If a tool name is not found in the registry
    """
    return [TOOL_REGISTRY[tool_name] for tool_name in tool_names]

def list_available_tools() -> List[str]:
    """
    List all available tools.
    
    Returns:
        List of tool names
    """
    return list(TOOL_REGISTRY.keys())
