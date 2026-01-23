# chess/tools.py
"""
Chess puzzle tools for AgentFly.

These tools allow agents to interact with chess puzzles:
- chess_move: Make a move on the board
- chess_get_state: Get the current board state
- chess_get_legal_moves: List all legal moves
"""

import traceback

from ...tool_base import tool
from ....envs.chess_env import ChessPuzzleEnv


@tool(
    env_cls=ChessPuzzleEnv,
    name="chess_move",
    description="Make a chess move in the current puzzle. The move can be in UCI format (e.g., 'e2e4', 'g1f3', 'e7e8q' for promotion) or standard algebraic notation (e.g., 'e4', 'Nf3', 'O-O' for castling, 'Qxf7#' for checkmate). Returns whether the move was correct and the new board state.",
    stateful=True,
    pool_size=8
)
async def chess_move(move: str, env: ChessPuzzleEnv):
    """
    Make a chess move in the puzzle.

    Args:
        move (str): The move to make. Can be in UCI format (e.g., 'e2e4', 'h5f7')
                   or SAN format (e.g., 'e4', 'Nf3', 'Qxf7+', 'O-O').
        env (ChessPuzzleEnv): The chess puzzle environment instance (auto-injected).

    Returns:
        str: The result of the move including:
             - Whether the move was correct for the puzzle
             - The new board state (FEN and visual representation)
             - Current game status (check, checkmate, etc.)
             - Error message if the move is invalid/illegal
    """
    try:
        result = await env.step(move)
        return result
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@tool(
    env_cls=ChessPuzzleEnv,
    name="chess_get_state",
    description="Get the current chess board state including FEN notation, visual board representation, whose turn it is, and puzzle status. Use this to understand the current position before making a move.",
    stateful=True,
    pool_size=8
)
async def chess_get_state(env: ChessPuzzleEnv):
    """
    Get the current state of the chess puzzle.

    Args:
        env (ChessPuzzleEnv): The chess puzzle environment instance (auto-injected).

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
        result = await env.step("get_state")
        return result
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@tool(
    env_cls=ChessPuzzleEnv,
    name="chess_get_legal_moves",
    description="Get all legal moves in the current position. Each move is shown in both UCI format (e.g., 'e2e4') and standard algebraic notation (e.g., 'e4'). Use this when you need to know what moves are available.",
    stateful=True,
    pool_size=8
)
async def chess_get_legal_moves(env: ChessPuzzleEnv):
    """
    Get all legal moves in the current position.

    Args:
        env (ChessPuzzleEnv): The chess puzzle environment instance (auto-injected).

    Returns:
        str: A comma-separated list of legal moves in format "uci (san)",
             e.g., "e2e4 (e4), g1f3 (Nf3), d2d4 (d4)"
             Sorted alphabetically by UCI notation.
    """
    try:
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
