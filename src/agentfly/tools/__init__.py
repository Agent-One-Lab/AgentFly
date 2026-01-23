from .decorator import tool
from .tool_base import BaseTool
from .src.alfworld.tools import (alfworld_get_admissible_commands,
                                 alfworld_get_task_objective, alfworld_reset,
                                 alfworld_step)
from .src.calculate.tools import calculator
from .src.code.tools import CodeInterpreterTool, code_interpreter
from .src.react.tools import answer_math, answer_qa
from .src.scienceworld.tools import scienceworld_explorer
from .src.search.async_dense_retriever import asyncdense_retrieve
from .src.search.dense_retriever import dense_retrieve
from .src.search.google_search import google_search_serper
from .src.ui.tools import pyautogui_code_generator
from .src.webshop.tools import webshop_browser
from .registry import get_tool_from_name, get_tools_from_names, register_tool


@tool()
def hallucination_tool(tool_name):
    return f"Hallucinated tool: {tool_name} does not exist."


@tool()
def invalid_input_tool(tool_input):
    return f"Invalid input: {tool_input}, input must be a valid JSON object."


__all__ = [
    "tool",
    "BaseTool",
    "code_interpreter",
    "CodeInterpreterTool",
    "alfworld_step",
    "alfworld_get_task_objective",
    "alfworld_get_admissible_commands",
    "alfworld_reset",
    "calculator",
    "google_search_serper",
    "dense_retrieve",
    "asyncdense_retrieve",
    "scienceworld_explorer",
    "webshop_browser",
    "answer_qa",
    "answer_math",
    "pyautogui_code_generator",
    "hallucination_tool",
    "invalid_input_tool",
    "get_tool_from_name",
    "get_tools_from_names",
    "register_tool",
]
