from typing import List
from ..tool_base import BaseTool


TOOL_REGISTRY = {}


def register_tool(tool_name, tool_func):
    """
    Register a tool in the tool registry.

    Args:
        tool_name: The name of the tool
        tool_func: The tool function or BaseTool instance
    """
    global TOOL_REGISTRY
    TOOL_REGISTRY[tool_name] = tool_func


def get_tool_from_name(tool_name: str) -> BaseTool:
    """
    Get a tool instance from its name.
    """
    return TOOL_REGISTRY[tool_name]


def get_tools_from_names(tool_names: List[str]) -> List[BaseTool]:
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
