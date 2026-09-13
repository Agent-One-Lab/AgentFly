"""Built-in tool/skill implementations, imported lazily.

These modules pull heavy, backend-specific dependencies (sympy, datasets, torch,
transformers, ray, ...) and register their tools via the ``@tool`` decorator as a
side effect of import. They are deliberately NOT imported by ``tools/__init__``
at package-load time — a bare ``import agentfly.tools`` (and therefore
``import agentfly.agents``) must stay cheap.

This module is imported on demand by ``tools/__init__.__getattr__`` (attribute
access like ``from agentfly.tools import alfworld_step``) and by
``tools.registry.registry.ensure_builtins_loaded`` (name lookup via
``get_tools_from_names`` / ``list_available_tools``). Importing it once registers
every built-in tool and binds every built-in name into this module's namespace,
which ``__init__`` re-exports.
"""

from .src.alfworld.tools import alfworld_step
from .src.calculate.tools import calculator
from .src.chess.tools import chess_get_legal_moves, chess_get_state, chess_move
from .src.code.tools import CodeInterpreterTool, code_interpreter
from .src.react.tools import answer_math, answer_qa
from .src.scienceworld.tools import scienceworld_explorer
from .src.search.async_dense_retrieve_api import async_dense_retrieve_api
from .src.search.async_dense_retriever import async_dense_retrieve
from .src.search.dense_retriever import dense_retrieve
# from .src.search.google_search import google_search_serper
from .src.ui.tools import pyautogui_code_generator
from .src.webshop.tools import webshop_browser
from .src.context.tools import summarize
from .src.file.tools import (
    create_file,
    grep_search,
    list_files,
    read_file,
    edit_file,
    run_python,
    undo_edit,
)
from .src.miniswe import miniswe_bash
from .src.shell.tools import run_shell_command
from .src.skills import (
    Skill,
    load_skill,
    load_skills,
    read_skill_file,
    run_skill_script,
)
