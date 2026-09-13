"""AgentFly tools package.

The light core (the ``@tool`` decorator, the registry accessors, ``BaseTool`` and
the result types) is imported eagerly. The built-in tool/skill *implementations*
live in :mod:`.impls` and are imported lazily — they pull heavy deps (sympy,
datasets, torch, ray, ...), so a bare ``import agentfly.tools`` (and hence
``import agentfly.agents``) stays cheap. The impls load on first access, whether
by attribute (``from agentfly.tools import alfworld_step``) or by name
(``get_tools_from_names`` / ``list_available_tools``).
"""

from typing import TYPE_CHECKING

from .decorator import tool
from .registry import get_tool_from_name, get_tools_from_names, register_tool
from .tool_base import BaseTool
from .types import ToolResult, ToolReturn

if TYPE_CHECKING:
    # Static tooling cannot discover exports through __getattr__. Keep these
    # declarations in sync with the valid lazy public exports in __all__;
    # google_search_serper is still unavailable in impls. This block does not
    # load implementations at runtime.
    from .impls import (
        CodeInterpreterTool,
        Skill,
        alfworld_step,
        answer_math,
        answer_qa,
        async_dense_retrieve,
        async_dense_retrieve_api,
        calculator,
        chess_get_legal_moves,
        chess_get_state,
        chess_move,
        code_interpreter,
        create_file,
        dense_retrieve,
        edit_file,
        grep_search,
        list_files,
        load_skill,
        load_skills,
        miniswe_bash,
        pyautogui_code_generator,
        read_file,
        read_skill_file,
        run_python,
        run_shell_command,
        run_skill_script,
        scienceworld_explorer,
        summarize,
        undo_edit,
        webshop_browser,
    )


def __getattr__(name):
    # Lazy attribute access for the built-in impls/classes/skills (everything in
    # the impls module). Route through the registry loader first so user-vs-builtin
    # name collisions resolve consistently (user wins), then read the name off it.
    # Guard dunder names and the loader module name so importing ``impls`` — which
    # itself resolves ``tools.impls`` — can't recurse back into here.
    if name == "impls" or name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .registry.registry import ensure_builtins_loaded

    ensure_builtins_loaded()
    from . import impls

    try:
        value = getattr(impls, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    globals()[name] = value  # cache so __getattr__ isn't hit again for this name
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "tool",
    "BaseTool",
    "ToolResult",
    "ToolReturn",
    "code_interpreter",
    "CodeInterpreterTool",
    "summarize",
    "alfworld_step",
    "calculator",
    "google_search_serper",
    "dense_retrieve",
    "async_dense_retrieve",
    "async_dense_retrieve_api",
    "scienceworld_explorer",
    "webshop_browser",
    "answer_qa",
    "answer_math",
    "pyautogui_code_generator",
    "get_tool_from_name",
    "get_tools_from_names",
    "register_tool",
    "chess_move",
    "chess_get_state",
    "chess_get_legal_moves",
    "create_file",
    "grep_search",
    "list_files",
    "read_file",
    "edit_file",
    "run_python",
    "undo_edit",
    "miniswe_bash",
    "run_shell_command",
    "Skill",
    "load_skill",
    "load_skills",
    "read_skill_file",
    "run_skill_script",
]
