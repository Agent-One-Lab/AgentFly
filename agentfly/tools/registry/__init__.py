from .registry import TOOL_REGISTRY

# from ..src.code.tools import code_interpreter
# from ..src.alfworld.tools import (
#     alfworld_step,
#     alfworld_get_task_objective,
#     alfworld_get_admissible_commands,
#     alfworld_reset
# )
# from ..src.calculate.tools import calculator
# from ..src.search.google_search import google_search_serper
# from ..src.search.dense_retriever import dense_retrieve
# from .src.search.async_dense_retriever import asyncdense_retrieve
# # from .src.search.http_retriever import http_retrieve
# from ..src.webshop.tools import webshop_browser
# from ..src.react.tools import answer_qa, answer_math
# from ..src.search.async_dense_retriever import asyncdense_retrieve
# from ..src.scienceworld.tools import scienceworld_explorer
# from ..src.ui.tools import pyautogui_code_generator
# from ..decorator import tool

# Global tool registry
# Add explicit tools in case they weren't auto-registered
# EXPLICIT_TOOLS = {
#     "asyncdense_retrieve": asyncdense_retrieve,
#     "dense_retrieve": dense_retrieve,
#     # "http_retrieve": http_retrieve,
#     "code_interpreter": code_interpreter,
#     "alfworld_step": alfworld_step,
#     "alfworld_reset": alfworld_reset,
#     "alfworld_get_task_objective": alfworld_get_task_objective,
#     "alfworld_get_admissible_commands": alfworld_get_admissible_commands,
#     "google_search": google_search_serper,
#     "answer_qa": answer_qa,
#     "answer_math": answer_math,
#     # "hallucination_tool": hallucination_tool,
#     # "invalid_input_tool": invalid_input_tool,
#     "dense_retrieve": dense_retrieve,
#     "pyautogui_code_generator": pyautogui_code_generator,
#     "calculator": calculator
# }

# # Update the registry with explicit tools
# for _name, _tool in EXPLICIT_TOOLS.items():
#     if _name not in TOOL_REGISTRY:
#         TOOL_REGISTRY[_name] = _tool
