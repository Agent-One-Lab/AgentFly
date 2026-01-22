from .decorator import tool
from .src.code.tools import code_interpreter, CodeInterpreterTool
from .src.alfworld.tools import (
    alfworld_step,
    alfworld_get_task_objective,
    alfworld_get_admissible_commands,
    alfworld_reset,
)
from .src.calculate.tools import calculator
from .src.search.google_search import google_search_serper
from .src.search.dense_retriever import dense_retrieve
from .src.search.async_dense_retriever import asyncdense_retrieve
from .src.scienceworld.tools import scienceworld_explorer
from .src.webshop.tools import webshop_browser
from .src.react.tools import answer_qa, answer_math
from .src.ui.tools import pyautogui_code_generator


@tool()
def hallucination_tool(tool_name):
    return f"Hallucinated tool: {tool_name} does not exist."


@tool()
def invalid_input_tool(tool_input):
    return f"Invalid input: {tool_input}, input must be a valid JSON object."


__all__ = [
    "tool",
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
]
