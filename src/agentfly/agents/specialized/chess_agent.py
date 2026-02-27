import logging
import os
import re
from typing import Any, List, Tuple

from ..agent_base import BaseAgent

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.environ.get("VERL_LOGGING_LEVEL", "INFO"))


CHESS_SYSTEM_PROMPT = (
    "Solve the given chess puzzle. You must conduct reasoning inside <think> and </think>"
    " first every time you get new information. After reasoning, you can use the following"
    " tools to interact with the board:\n\n"
    "- <chess_get_state> </chess_get_state>: Get the current board state including FEN,"
    " visual board, whose turn it is, and puzzle status.\n"
    "- <chess_get_legal_moves> </chess_get_legal_moves>: List all legal moves in the"
    " current position.\n"
    "- <chess_move> move </chess_move>: Make a move on the board. The move can be in UCI"
    " format (e.g., 'e2e4') or standard algebraic notation (e.g., 'Nf3').\n\n"
    "You should start by calling <chess_get_state> </chess_get_state> to see the current"
    " position. You may call <chess_get_legal_moves> </chess_get_legal_moves> to see all"
    " available moves. When you have decided on the best move, play it with"
    " <chess_move> move </chess_move>. Continue until the puzzle is solved. When the puzzle"
    " is solved, provide your final answer inside <answer> and </answer> listing the moves"
    " you played. For example, <answer> e2e4 d7d5 </answer>."
)


def parse_chess_response(response: str) -> Tuple[str, str, dict]:
    """
    Parse a chess agent response and extract thinking, answer, and a single tool call.

    The chess agent uses XML-style tags for tool invocation:
      - <chess_get_state> </chess_get_state>
      - <chess_get_legal_moves> </chess_get_legal_moves>
      - <chess_move> move </chess_move>

    Args:
        response: Raw model output string.

    Returns:
        Tuple of (thinking, answer, tool_call_dict_or_None).
        tool_call_dict has keys "name" and "arguments" when present.
    """
    LOGGER.debug(f"Response: {response}")

    # Extract thinking
    thinking = ""
    try:
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
    except Exception as e:
        LOGGER.debug(f"Error parsing thinking: {e}")

    # Extract answer
    answer = ""
    try:
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
    except Exception as e:
        LOGGER.debug(f"Error parsing answer: {e}")

    # Extract tool calls — we take the first one found
    tool_call = None
    try:
        # chess_get_state (no arguments)
        if re.search(r"<chess_get_state>\s*</chess_get_state>", response, re.DOTALL):
            tool_call = {"name": "chess_get_state", "arguments": "{}"}

        # chess_get_legal_moves (no arguments)
        elif re.search(
            r"<chess_get_legal_moves>\s*</chess_get_legal_moves>", response, re.DOTALL
        ):
            tool_call = {"name": "chess_get_legal_moves", "arguments": "{}"}

        # chess_move (has a move argument)
        else:
            move_match = re.search(
                r"<chess_move>\s*(.*?)\s*</chess_move>", response, re.DOTALL
            )
            if move_match:
                move = move_match.group(1).strip()
                tool_call = {
                    "name": "chess_move",
                    "arguments": f'{{"move": "{move}"}}',
                }
    except Exception as e:
        LOGGER.debug(f"Error parsing tool call: {e}")

    LOGGER.debug(f"Thinking: {thinking}")
    LOGGER.debug(f"Answer: {answer}")
    LOGGER.debug(f"Tool call: {tool_call}")

    return thinking, answer, tool_call


class ChessAgent(BaseAgent):
    """
    Specialized agent for solving chess puzzles.

    Parses XML-style tool tags (<chess_get_state>, <chess_get_legal_moves>,
    <chess_move>) from the model output and converts them into the standard
    tool_calls format expected by the framework.
    """

    def __init__(
        self,
        model_name_or_path: str,
        system_prompt: str = None,
        tools: List = None,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            system_prompt=system_prompt or CHESS_SYSTEM_PROMPT,
            tools=tools,
            **kwargs,
        )

    def parse(self, responses: List[str], tools: List[Any] = None) -> List[dict]:
        """
        Parse model responses and extract chess tool calls.

        Returns:
            List of message dicts with standard tool_calls format.
        """
        parsed = [parse_chess_response(response) for response in responses]

        new_messages_list = []
        for i, (thinking, answer, tool_call) in enumerate(parsed):
            tool_calls = []
            if tool_call is not None:
                tool_calls.append(
                    {
                        "id": None,
                        "type": "function",
                        "function": tool_call,
                    }
                )

            message = {
                "role": "assistant",
                "content": [{"type": "text", "text": responses[i]}],
                "tool_calls": tool_calls,
                "loss": True,
                "status": "continue" if len(tool_calls) > 0 else "terminal",
            }
            new_messages_list.append(message)

        return new_messages_list
