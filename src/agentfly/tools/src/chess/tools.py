# chess/tools.py
"""
Chess puzzle tools for AgentFly.

These tools allow agents to interact with chess puzzles:
- chess_move: Make a move on the board
- chess_get_state: Get the current board state
- chess_get_legal_moves: List all legal moves

Each tool acquires the puzzle env via the rollout context, sharing the same
resource the chess reward uses.
"""

import traceback

from ....core import Context
from ....envs.chess_env import ChessPuzzleSpec
from ...decorator import tool


async def _get_chess_env(context: Context):
    """Acquire the chess puzzle resource and reset it once per rollout.

    The puzzle parameters (``puzzle_id``, ``fen``, ``moves``) are read from
    ``context.metadata`` on first acquire so each rollout gets the puzzle
    that was attached to its dataset row.
    """
    need_reset = not context.is_spec_acquired(ChessPuzzleSpec)
    env = await context.acquire_resource(
        spec=ChessPuzzleSpec,
        scope="global",
        backend="local",
    )
    if need_reset:
        meta = context.metadata or {}
        env_args = {
            k: meta[k] for k in ("puzzle_id", "fen", "moves") if k in meta
        }
        if env_args:
            await env.reset(env_args=env_args)
        else:
            await env.reset()
    return env


@tool(
    name="chess_move",
    description="Make a chess move in the current puzzle. The move can be in UCI format (e.g., 'e2e4', 'g1f3', 'e7e8q' for promotion) or standard algebraic notation (e.g., 'e4', 'Nf3', 'O-O' for castling, 'Qxf7#' for checkmate). Returns whether the move was correct and the new board state.",
    stateful=True,
)
async def chess_move(move: str, context: Context):
    """
    Make a chess move in the puzzle.

    Args:
        move (str): The move to make. Can be in UCI format (e.g., 'e2e4', 'h5f7')
                   or SAN format (e.g., 'e4', 'Nf3', 'Qxf7+', 'O-O').
        context (Context): Injected rollout context; used to acquire the chess puzzle resource.

    Returns:
        str: The result of the move including:
             - Whether the move was correct for the puzzle
             - The new board state (FEN and visual representation)
             - Current game status (check, checkmate, etc.)
             - Error message if the move is invalid/illegal
    """
    try:
        env = await _get_chess_env(context)
        result = await env.step(move)
        return result
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="chess_get_state",
    description="Get the current chess board state including FEN notation, visual board representation, whose turn it is, and puzzle status. Use this to understand the current position before making a move.",
    stateful=True,
)
async def chess_get_state(context: Context):
    """
    Get the current state of the chess puzzle.

    Args:
        context (Context): Injected rollout context; used to acquire the chess puzzle resource.

    Returns:
        str: A detailed representation of the current board state including:
             - FEN notation (standard chess position encoding)
             - ASCII board visualization
             - Whose turn it is (White or Black)
             - Number of legal moves available
             - Check/checkmate/stalemate status
             - Whether the puzzle is solved
             - Moves played so far
    """
    try:
        env = await _get_chess_env(context)
        result = await env.step("get_state")
        return result
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="chess_get_legal_moves",
    description="Get all legal moves in the current position. Each move is shown in both UCI format (e.g., 'e2e4') and standard algebraic notation (e.g., 'e4'). Use this when you need to know what moves are available.",
    stateful=True,
)
async def chess_get_legal_moves(context: Context):
    """
    Get all legal moves in the current position.

    Args:
        context (Context): Injected rollout context; used to acquire the chess puzzle resource.

    Returns:
        str: A comma-separated list of legal moves in format "uci (san)",
             e.g., "e2e4 (e4), g1f3 (Nf3), d2d4 (d4)"
             Sorted alphabetically by UCI notation.
    """
    try:
        env = await _get_chess_env(context)
        result = await env.step("get_legal_moves")
        return result
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    print("Chess Tools Schemas:")
    print("=" * 50)
    print("\nchess_move schema:")
    print(chess_move.schema)
    print("\nchess_get_state schema:")
    print(chess_get_state.schema)
    print("\nchess_get_legal_moves schema:")
    print(chess_get_legal_moves.schema)
