# chess_env.py
"""
Chess puzzle environment using python-chess and Stockfish engine.
Unlike Docker-based environments, this runs locally with python-chess library.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine

from .env_base import BaseEnv


class ChessPuzzleEnv(BaseEnv):
    """
    Chess puzzle environment using python-chess and Stockfish.

    This is a non-Docker environment that runs locally with:
    - python-chess for board state management and move validation
    - Stockfish engine for position evaluation and best move analysis

    Puzzle Format (Lichess-style):
    - FEN: starting position
    - Moves: solution moves in UCI format (e.g., "e2e4 e7e5 g1f3")
    - The first move is the opponent's move that sets up the puzzle
    - Subsequent moves are the solution the agent must find

    Attributes:
        stockfish_path (str): Path to the Stockfish binary
        analysis_time (float): Time in seconds for Stockfish analysis per position
        analysis_depth (int): Depth for Stockfish analysis
        max_moves (int): Maximum moves allowed per puzzle

    Example:
        ```python
        env = ChessPuzzleEnv()
        await env.start()
        obs = await env.reset({
            "puzzle_id": "test1",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "moves": "h5f7"
        })
        result = await env.step("Qxf7")  # or "h5f7" in UCI
        await env.aclose()
        ```
    """

    def __init__(
        self,
        stockfish_path: str = "/opt/homebrew/bin/stockfish",
        analysis_time: float = 0.1,
        analysis_depth: int = 20,
        max_moves: int = 20,
    ):
        """
        Initialize the chess puzzle environment.

        Args:
            stockfish_path: Path to the Stockfish binary. Common paths:
                - macOS (Homebrew): /opt/homebrew/bin/stockfish
                - Linux: /usr/bin/stockfish or /usr/games/stockfish
                - Windows: C:\\path\\to\\stockfish.exe
            analysis_time: Time in seconds for each Stockfish analysis
            analysis_depth: Depth for Stockfish search (higher = stronger but slower)
            max_moves: Maximum number of moves allowed per puzzle
        """
        super().__init__()
        self.stockfish_path = stockfish_path
        self.analysis_time = analysis_time
        self.analysis_depth = analysis_depth
        self.max_moves = max_moves

        # Engine and board state
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._board: Optional[chess.Board] = None

        # Puzzle state
        self._puzzle_id: Optional[str] = None
        self._puzzle_fen: Optional[str] = None
        self._solution_moves: List[str] = []
        self._current_solution_idx: int = 0
        self._moves_made: List[str] = []
        self._is_solved: bool = False

    async def start(self) -> None:
        """
        Start the Stockfish engine process.

        Unlike Docker-based environments, this simply spawns the Stockfish
        subprocess using python-chess's engine API.

        Raises:
            FileNotFoundError: If Stockfish binary is not found at the specified path
            chess.engine.EngineTerminatedError: If the engine fails to start
        """
        loop = asyncio.get_running_loop()
        self._engine = await loop.run_in_executor(
            None, chess.engine.SimpleEngine.popen_uci, self.stockfish_path
        )
        self._board = chess.Board()

    async def reset(self, env_args: Optional[Dict[str, Any]] = None) -> str:
        """
        Reset to a new puzzle.

        Args:
            env_args: Dictionary with puzzle data:
                - puzzle_id (str): Unique puzzle identifier
                - fen (str): Starting FEN position
                - moves (str): Space-separated UCI moves for the solution.
                  First move is opponent's setup move (auto-played).

        Returns:
            Initial observation (board state as text)

        Example:
            ```python
            obs = await env.reset({
                "puzzle_id": "abc123",
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                "moves": "h5f7"  # White mates with Qxf7#
            })
            ```
        """
        if env_args is None:
            # Default puzzle for testing: Scholar's mate position
            env_args = {
                "puzzle_id": "default",
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                "moves": "h5f7",
            }

        self._puzzle_id = env_args.get("puzzle_id", "unknown")
        self._puzzle_fen = env_args["fen"]
        self._solution_moves = env_args.get("moves", "").split()
        self._current_solution_idx = 0
        self._moves_made = []
        self._is_solved = False

        # Set up the board
        self._board = chess.Board(self._puzzle_fen)

        # In Lichess puzzles, the first move is typically the opponent's move
        # that sets up the puzzle. For single-move puzzles (like mate-in-1),
        # we don't auto-play the first move since the agent needs to find it.
        # We'll detect this by checking if the solution has more than 1 move.
        if len(self._solution_moves) > 1:
            # Play the setup move (opponent's move)
            try:
                setup_move = chess.Move.from_uci(self._solution_moves[0])
                if setup_move in self._board.legal_moves:
                    self._board.push(setup_move)
                    self._current_solution_idx = 1
            except ValueError:
                pass  # Invalid move format, skip

        return self._get_observation()

    async def step(self, action: str) -> Union[str, Dict[str, Any]]:
        """
        Execute an action in the chess environment.

        Args:
            action: Either:
                - A chess move in UCI format (e.g., "e2e4") or SAN (e.g., "e4", "Nf3")
                - "get_state" to get current board state
                - "get_legal_moves" to list all legal moves
                - "get_reward" to get current evaluation

        Returns:
            Observation string with result of the action, or dict for get_reward
        """
        action = action.strip()
        action_lower = action.lower()

        if action_lower == "get_state":
            return self._get_observation()

        if action_lower == "get_legal_moves":
            return self._get_legal_moves_text()

        if action_lower == "get_reward":
            return await self._get_evaluation()

        # Try to parse and make the move
        return await self._make_move(action)

    async def _make_move(self, move_str: str) -> str:
        """Process a move from the agent."""
        if self._is_solved:
            return "Puzzle already solved! No more moves needed."

        if self._board.is_game_over():
            return f"Game is over. Result: {self._board.result()}"

        if len(self._moves_made) >= self.max_moves:
            return f"Maximum moves ({self.max_moves}) reached."

        # Parse the move (try UCI first, then SAN)
        move = self._parse_move(move_str)
        if move is None:
            return f"Invalid move format: '{move_str}'. Legal moves: {self._get_legal_moves_text()}"

        if move not in self._board.legal_moves:
            return f"Illegal move: '{move_str}'. Legal moves: {self._get_legal_moves_text()}"

        # Make the move
        san_move = self._board.san(move)  # Get SAN before pushing
        self._board.push(move)
        self._moves_made.append(move.uci())

        # Check if this matches the solution
        expected_move = None
        if self._current_solution_idx < len(self._solution_moves):
            expected_move = self._solution_moves[self._current_solution_idx]

        is_correct = expected_move and move.uci() == expected_move

        if is_correct:
            self._current_solution_idx += 1

            # Check if puzzle is solved
            if self._current_solution_idx >= len(self._solution_moves):
                self._is_solved = True
                return (
                    f"Correct! {san_move} - Puzzle solved!\n\n{self._get_observation()}"
                )

            # Play the opponent's response (next move in solution)
            if self._current_solution_idx < len(self._solution_moves):
                try:
                    response_uci = self._solution_moves[self._current_solution_idx]
                    response_move = chess.Move.from_uci(response_uci)
                    if response_move in self._board.legal_moves:
                        response_san = self._board.san(response_move)
                        self._board.push(response_move)
                        self._current_solution_idx += 1

                        # Check again if solved after response
                        if self._current_solution_idx >= len(self._solution_moves):
                            self._is_solved = True
                            return f"Correct! {san_move}\nOpponent played: {response_san}\nPuzzle solved!\n\n{self._get_observation()}"

                        return f"Correct! {san_move}\nOpponent played: {response_san}\nYour turn to continue.\n\n{self._get_observation()}"
                except ValueError:
                    pass

            return f"Correct! {san_move}\n\n{self._get_observation()}"
        else:
            # Wrong move - still allow it but note it's not the solution
            return f"Move played: {san_move} (not the puzzle solution)\n\n{self._get_observation()}"

    def _parse_move(self, move_str: str) -> Optional[chess.Move]:
        """Parse a move string in UCI or SAN format."""
        move_str = move_str.strip()

        # Try UCI format first (e.g., "e2e4", "e7e8q" for promotion)
        try:
            return chess.Move.from_uci(move_str.lower())
        except ValueError:
            pass

        # Try SAN format (e.g., "e4", "Nf3", "O-O", "Qxf7#")
        try:
            return self._board.parse_san(move_str)
        except ValueError:
            pass

        return None

    def _get_observation(self) -> str:
        """Generate a text observation of the current board state."""
        parts = [
            f"FEN: {self._board.fen()}",
            "",
            str(self._board),  # ASCII board representation
            "",
            f"Turn: {'White' if self._board.turn else 'Black'}",
            f"Legal moves: {len(list(self._board.legal_moves))} available",
        ]

        if self._board.is_check():
            parts.append("Status: CHECK!")

        if self._is_solved:
            parts.append("Status: PUZZLE SOLVED!")
        elif self._board.is_checkmate():
            winner = "Black" if self._board.turn else "White"
            parts.append(f"Status: CHECKMATE! {winner} wins.")
        elif self._board.is_stalemate():
            parts.append("Status: STALEMATE! Draw.")
        elif self._board.is_insufficient_material():
            parts.append("Status: Draw by insufficient material.")

        parts.append(
            f"\nMoves played: {', '.join(self._moves_made) if self._moves_made else 'None'}"
        )

        return "\n".join(parts)

    def _get_legal_moves_text(self) -> str:
        """Get legal moves as a formatted string."""
        moves = []
        for move in self._board.legal_moves:
            san = self._board.san(move)
            moves.append(f"{move.uci()} ({san})")
        return ", ".join(sorted(moves)) if moves else "No legal moves"

    async def _get_evaluation(self) -> Dict[str, Any]:
        """Get Stockfish evaluation of current position."""
        if self._engine is None:
            return {"observation": "Engine not available", "reward": 0.0}

        loop = asyncio.get_running_loop()
        try:
            info = await loop.run_in_executor(
                None,
                lambda: self._engine.analyse(
                    self._board,
                    chess.engine.Limit(
                        time=self.analysis_time, depth=self.analysis_depth
                    ),
                ),
            )

            score = info.get("score")
            if score:
                pov_score = score.white() if self._board.turn else score.black()
                cp = pov_score.score(mate_score=10000)

                if cp is not None:
                    # Normalize to 0-1 range (sigmoid-like)
                    normalized = max(0.0, min(1.0, (cp + 500) / 1000))
                    return {
                        "observation": f"Evaluation: {cp / 100:.2f} pawns (centipawns: {cp})",
                        "reward": normalized,
                        "centipawns": cp,
                        "is_solved": self._is_solved,
                    }
                else:
                    mate = pov_score.mate()
                    if mate is not None:
                        reward = 1.0 if mate > 0 else 0.0
                        return {
                            "observation": f"Mate in {abs(mate)}"
                            if mate > 0
                            else f"Getting mated in {abs(mate)}",
                            "reward": reward,
                            "mate_in": mate,
                            "is_solved": self._is_solved,
                        }
        except Exception as e:
            return {"observation": f"Evaluation error: {e}", "reward": 0.0}

        return {"observation": "Unable to evaluate position", "reward": 0.0}

    async def get_best_move(self) -> Tuple[str, int]:
        """
        Get the best move according to Stockfish with evaluation.

        Returns:
            Tuple of (best_move_uci, centipawns)
        """
        if self._engine is None:
            return ("", 0)

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._engine.analyse(
                    self._board,
                    chess.engine.Limit(
                        time=self.analysis_time, depth=self.analysis_depth
                    ),
                ),
            )

            best_move = result.get("pv", [None])[0]
            score = result.get("score")

            cp = 0
            if score:
                pov_score = score.white() if self._board.turn else score.black()
                cp = pov_score.score(mate_score=10000) or 0

            return (best_move.uci() if best_move else "", cp)
        except Exception:
            return ("", 0)

    async def evaluate_move(self, move_uci: str) -> int:
        """
        Evaluate a specific move by comparing position before and after.

        Args:
            move_uci: Move in UCI format to evaluate

        Returns:
            Centipawn difference (positive = good move, negative = bad move)
        """
        if self._engine is None:
            return 0

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return -10000  # Invalid move

        if move not in self._board.legal_moves:
            return -10000  # Illegal move penalty

        loop = asyncio.get_running_loop()

        try:
            # Get best move evaluation before this move
            best_move_uci, best_cp = await self.get_best_move()

            # Make the move temporarily
            self._board.push(move)

            # Evaluate position after move (from opponent's perspective, so negate)
            after_info = await loop.run_in_executor(
                None,
                lambda: self._engine.analyse(
                    self._board,
                    chess.engine.Limit(
                        time=self.analysis_time, depth=self.analysis_depth
                    ),
                ),
            )

            # Undo the move
            self._board.pop()

            after_score = after_info.get("score")
            if after_score:
                # Get score from the perspective of the player who just moved
                pov_score = (
                    after_score.black() if self._board.turn else after_score.white()
                )
                after_cp = pov_score.score(mate_score=10000) or 0
            else:
                after_cp = 0

            # If this was the best move, return 0 (no loss)
            if move_uci == best_move_uci:
                return 0

            # Return centipawn loss (negative = worse than best move)
            return after_cp - best_cp

        except Exception:
            return 0

    @property
    def is_solved(self) -> bool:
        """Whether the puzzle has been solved correctly."""
        return self._is_solved

    @property
    def puzzle_id(self) -> str:
        """The current puzzle's ID."""
        return self._puzzle_id or ""

    @property
    def moves_made(self) -> List[str]:
        """List of moves made by the agent (UCI format)."""
        return self._moves_made.copy()

    @property
    def board(self) -> chess.Board:
        """The current chess board state."""
        return self._board

    async def aclose(self) -> None:
        """
        Close the Stockfish engine and release resources.
        """
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def close(self) -> None:
        """
        Synchronous close - quit the engine.
        """
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    @staticmethod
    async def acquire():
        """
        Factory method to create and start a chess environment.

        Returns:
            ChessPuzzleEnv: A fully initialized environment ready for use

        Example:
            ```python
            env = await ChessPuzzleEnv.acquire()
            obs = await env.reset({"fen": "...", "moves": "..."})
            await env.aclose()
            ```
        """
        env = ChessPuzzleEnv()
        await env.start()
        return env
