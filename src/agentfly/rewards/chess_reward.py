# chess_reward.py
"""
Chess puzzle reward functions for AgentFly.

Provides two reward functions:
- chess_puzzle_reward: Dense reward based on Stockfish evaluation (move quality)
- chess_puzzle_reward_simple: Binary reward (solved/not solved)
"""

from typing import Any, Dict

from ..envs.chess_env import ChessPuzzleEnv
from .reward_base import reward


@reward(name="chess_puzzle_reward", env_cls=ChessPuzzleEnv, pool_size=8)
async def chess_puzzle_reward(prediction: str, env: ChessPuzzleEnv) -> Dict[str, Any]:
    """
    Calculate reward for chess puzzle solving based on Stockfish evaluation.

    This reward function provides:
    1. Dense reward based on move quality (centipawn evaluation)
    2. Bonus for solving the puzzle correctly
    3. Penalty for making suboptimal moves

    The reward is structured to encourage:
    - Finding the best moves (matching Stockfish recommendations)
    - Solving puzzles completely
    - Making progress even with imperfect moves

    Args:
        prediction (str): The agent's final response/output (not used directly).
        env (ChessPuzzleEnv): The chess puzzle environment instance.

    Returns:
        dict: A dictionary containing:
            - reward (float): The calculated reward value (0.0 to 1.0+)
            - is_solved (bool): Whether the puzzle was solved correctly
            - moves_made (int): Number of moves made
            - best_move_matches (int): How many moves matched Stockfish's best move
            - centipawn_score (float): Average centipawn quality of moves (0-100 scale)
            - output (str): Human-readable summary
    """
    # Get puzzle state
    is_solved = env.is_solved
    moves_made = env.moves_made
    num_moves = len(moves_made)

    # Calculate solve bonus
    if is_solved:
        solve_reward = 1.0
    else:
        # Partial credit for progress through the solution
        solution_len = len(env._solution_moves)
        if solution_len > 1:
            # Adjust for the setup move
            progress = max(0, env._current_solution_idx - 1) / (solution_len - 1)
            solve_reward = progress * 0.5  # Up to 0.5 for partial progress
        elif solution_len == 1:
            # Single move puzzle
            solve_reward = 0.0
        else:
            solve_reward = 0.0

    # Calculate move quality reward using Stockfish
    centipawn_total = 0.0
    best_move_matches = 0

    if num_moves > 0 and env._engine is not None:
        # Evaluate each move made
        # We need to replay from the starting position
        import chess
        temp_board = chess.Board(env._puzzle_fen)

        # Apply setup move if it was made
        if len(env._solution_moves) > 1 and env._current_solution_idx >= 1:
            try:
                setup_move = chess.Move.from_uci(env._solution_moves[0])
                if setup_move in temp_board.legal_moves:
                    temp_board.push(setup_move)
            except ValueError:
                pass

        for i, move_uci in enumerate(moves_made):
            try:
                # Get best move for this position
                best_move, best_cp = await env.get_best_move()

                # Check if agent's move matches best move
                if move_uci == best_move:
                    best_move_matches += 1
                    centipawn_total += 100.0  # Perfect score for matching best
                else:
                    # Evaluate the quality of the actual move
                    cp_loss = await env.evaluate_move(move_uci)
                    # Convert centipawn loss to 0-100 scale
                    # 0 cp loss = 100, -300 cp loss = 0
                    normalized = max(0.0, min(100.0, 100.0 + (cp_loss / 3.0)))
                    centipawn_total += normalized

                # Apply the move to continue analysis
                move = chess.Move.from_uci(move_uci)
                if move in temp_board.legal_moves:
                    temp_board.push(move)

            except Exception:
                # If analysis fails, give partial credit
                centipawn_total += 50.0

    # Average centipawn score
    avg_cp = centipawn_total / num_moves if num_moves > 0 else 50.0
    move_quality_reward = avg_cp / 100.0  # 0.0 to 1.0

    # Combine rewards
    # 60% for solving, 40% for move quality
    total_reward = 0.6 * solve_reward + 0.4 * move_quality_reward

    # Build output summary
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


@reward(name="chess_puzzle_reward_simple", env_cls=ChessPuzzleEnv, pool_size=8)
async def chess_puzzle_reward_simple(
    prediction: str, env: ChessPuzzleEnv
) -> Dict[str, Any]:
    """
    Simple binary reward for chess puzzle solving.

    Returns 1.0 if puzzle is solved correctly, 0.0 otherwise.
    Useful for comparison with dense reward and for simpler training setups
    where you only care about correct solutions.

    Args:
        prediction (str): The agent's final response/output (not used).
        env (ChessPuzzleEnv): The chess puzzle environment instance.

    Returns:
        dict: Contains:
            - reward (float): 1.0 if solved, 0.0 otherwise
            - is_solved (bool): Whether the puzzle was solved
            - output (str): Human-readable status message
    """
    is_solved = env.is_solved

    return {
        "reward": 1.0 if is_solved else 0.0,
        "is_solved": is_solved,
        "output": f"Puzzle {'solved' if is_solved else 'not solved'}",
    }
