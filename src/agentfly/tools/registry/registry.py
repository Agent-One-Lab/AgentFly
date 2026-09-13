from typing import List

from ..tool_base import BaseTool

TOOL_REGISTRY = {}

builtins_loaded = False


def ensure_builtins_loaded() -> None:
    """Import the built-in tool impls once so name lookups resolve.

    The built-ins live in :mod:`agentfly.tools.impls` and register themselves on
    import (heavy deps), so we defer that until a name is actually looked up.
    User tools registered earlier via ``@tool`` are preserved: importing the
    built-ins only ADDS to ``TOOL_REGISTRY``, and if a built-in shares a name with
    a user tool, the user's entry is re-applied on top so the user still wins
    (matching the old import-order behavior).
    """
    global builtins_loaded
    if builtins_loaded:
        return
    user_entries = dict(TOOL_REGISTRY)  # snapshot user regs (collision: user wins)
    builtins_loaded = True  # set before import so re-entrant lookups are a no-op
    from .. import impls  # noqa: F401  (imports every built-in impl -> registers)

    TOOL_REGISTRY.update(user_entries)


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
    ensure_builtins_loaded()
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
    from ...utils.references import resolve_reference
    ensure_builtins_loaded()
    # Each entry is a registered name OR an import reference
    # ("module:tool", "/path.py:tool"); names and refs may be mixed.
    return [resolve_reference(name, TOOL_REGISTRY, "tool") for name in tool_names]


def list_available_tools() -> List[str]:
    """
    List all available tools.

    Returns:
        List of tool names
    """
    ensure_builtins_loaded()
    return list(TOOL_REGISTRY.keys())
