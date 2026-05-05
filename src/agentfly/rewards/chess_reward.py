# chess_reward.py
"""
Chess puzzle reward functions for AgentFly.

Provides two reward functions:
- chess_puzzle_reward: Dense reward based on Stockfish evaluation (move quality)
- chess_puzzle_reward_simple: Binary reward (solved/not solved)

Both rewards acquire the puzzle environment via the rollout context, sharing
the same resource the chess tools use.
"""

from typing import Any, Dict
from ..core import Context
from ..envs.chess_env import ChessPuzzleEnv, ChessPuzzleSpec
from .reward_base import reward


@reward(name="chess_puzzle_reward")
async def chess_puzzle_reward(final_response: str, context: Context) -> Dict[str, Any]:
    """
    Calculate reward for chess puzzle solving based on Stockfish evaluation.

    This reward function provides:
    1. Dense reward based on move quality (centipawn evaluation)
    2. Bonus for solving the puzzle correctly
    3. Penalty for making suboptimal moves

    Args:
        final_response (str): The agent's final response/output (not used directly).
        context (Context): Injected rollout context; used to acquire the chess puzzle resource.

    Returns:
        dict: A dictionary containing:
            - reward (float): The calculated reward value (0.0 to 1.0+)
            - is_solved (bool): Whether the puzzle was solved correctly
            - moves_made (int): Number of moves made
            - best_move_matches (int): How many moves matched Stockfish's best move
            - centipawn_score (float): Average centipawn quality of moves (0-100 scale)
            - output (str): Human-readable summary
    """
    env: ChessPuzzleEnv = await context.acquire_resource(
        spec=ChessPuzzleSpec, scope="global", backend="local"
    )

    is_solved = env.is_solved
    moves_made = env.moves_made
    num_moves = len(moves_made)

    if is_solved:
        solve_reward = 1.0
    else:
        solution_len = len(env._solution_moves)
        if solution_len > 1:
            progress = max(0, env._current_solution_idx - 1) / (solution_len - 1)
            solve_reward = progress * 0.5
        elif solution_len == 1:
            solve_reward = 0.0
        else:
            solve_reward = 0.0

    centipawn_total = 0.0
    best_move_matches = 0

    if num_moves > 0 and env._engine is not None:
        import chess

        temp_board = chess.Board(env._puzzle_fen)

        if len(env._solution_moves) > 1 and env._current_solution_idx >= 1:
            try:
                setup_move = chess.Move.from_uci(env._solution_moves[0])
                if setup_move in temp_board.legal_moves:
                    temp_board.push(setup_move)
            except ValueError:
                pass

        for move_uci in moves_made:
            try:
                best_move, _ = await env.get_best_move()

                if move_uci == best_move:
                    best_move_matches += 1
                    centipawn_total += 100.0
                else:
                    cp_loss = await env.evaluate_move(move_uci)
                    normalized = max(0.0, min(100.0, 100.0 + (cp_loss / 3.0)))
                    centipawn_total += normalized

                move = chess.Move.from_uci(move_uci)
                if move in temp_board.legal_moves:
                    temp_board.push(move)

            except Exception:
                centipawn_total += 50.0

    avg_cp = centipawn_total / num_moves if num_moves > 0 else 50.0
    move_quality_reward = avg_cp / 100.0

    # 60% for solving, 40% for move quality
    total_reward = 0.6 * solve_reward + 0.4 * move_quality_reward

    output_parts = [
        f"Puzzle {'SOLVED!' if is_solved else 'not solved'}",
        f"Moves made: {num_moves}",
        f"Best move matches: {best_move_matches}/{num_moves}"
        if num_moves > 0
        else "No moves made",
        f"Average move quality: {avg_cp:.1f}/100",
        f"Total reward: {total_reward:.3f}",
    ]

    return {
        "reward": total_reward,
        "is_solved": is_solved,
        "moves_made": num_moves,
        "best_move_matches": best_move_matches,
        "centipawn_score": avg_cp,
        "output": "\n".join(output_parts),
    }


@reward(name="chess_puzzle_reward_simple")
async def chess_puzzle_reward_simple(
    final_response: str, context: Context
) -> Dict[str, Any]:
    """
    Simple binary reward for chess puzzle solving.

    Returns 1.0 if puzzle is solved correctly, 0.0 otherwise.

    Args:
        final_response (str): The agent's final response/output (not used).
        context (Context): Injected rollout context; used to acquire the chess puzzle resource.

    Returns:
        dict: Contains:
            - reward (float): 1.0 if solved, 0.0 otherwise
            - is_solved (bool): Whether the puzzle was solved
            - output (str): Human-readable status message
    """
    env: ChessPuzzleEnv = await context.acquire_resource(
        spec=ChessPuzzleSpec, scope="global", backend="local"
    )
    is_solved = env.is_solved

    return {
        "reward": 1.0 if is_solved else 0.0,
        "is_solved": is_solved,
        "output": f"Puzzle {'solved' if is_solved else 'not solved'}",
    }
