

from typing import List
from .tool_base import (
    BaseTool,
    submit_tool_call
)
from .decorator import tool
from .src.code.tools import code_interpreter, CodeInterpreterTool
from .src.alfworld.tools import (
    alfworld_step,
    alfworld_get_task_objective,
    alfworld_get_admissible_commands,
    alfworld_reset
)
from .src.calculate.tools import calculator
from .src.search.google_search import google_search_serper
from .src.search.dense_retriever import dense_retrieve
from .src.search.async_dense_retriever import asyncdense_retrieve
from .src.scienceworld.tools import scienceworld_explorer
# from .src.search.http_retriever import http_retrieve
from .src.webshop.tools import webshop_browser
from .src.react.tools import answer_qa, answer_math
from .src.ui.tools import pyautogui_code_generator
from .registry import TOOL_REGISTRY
from .registry import (
    register_tool,
    get_tool_from_name,
    get_tools_from_names,
    list_available_tools,
)

@tool()
def hallucination_tool(tool_name):
    return f"Hallucinated tool: {tool_name} does not exist."

@tool()
def invalid_input_tool(tool_input):
    return f"Invalid input: {tool_input}, input must be a valid JSON object."
