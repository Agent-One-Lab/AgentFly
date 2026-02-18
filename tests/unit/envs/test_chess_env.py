# test_chess_env.py
"""
Unit tests for the chess puzzle environment.

Note: These tests require Stockfish to be installed.
Install with: brew install stockfish (macOS) or apt-get install stockfish (Linux)
"""

import pytest
import pytest_asyncio
from agentfly.envs.chess_env import ChessPuzzleEnv


# Skip all tests if Stockfish is not available
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("chess"),
    reason="python-chess not installed"
)


@pytest_asyncio.fixture
async def chess_env():
    """Create and start a chess environment for testing."""
    env = ChessPuzzleEnv()
    try:
        await env.start()
        yield env
    finally:
        await env.aclose()


@pytest.mark.asyncio
async def test_env_start_and_close():
    """Test that environment can start and close properly."""
    env = ChessPuzzleEnv()
    await env.start()
    assert env._engine is not None
    assert env._board is not None
    await env.aclose()
    assert env._engine is None


@pytest.mark.asyncio
async def test_env_reset_default_puzzle(chess_env):
    """Test resetting to the default puzzle."""
    obs = await chess_env.reset()

    # Should have board state in observation
    assert "FEN:" in obs
    assert "Turn:" in obs
    assert "Legal moves:" in obs


@pytest.mark.asyncio
async def test_env_reset_custom_puzzle(chess_env):
    """Test resetting with a custom puzzle."""
    puzzle = {
        "puzzle_id": "test_mate_in_1",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "moves": "h5f7"  # Qxf7# is mate
    }

    obs = await chess_env.reset(puzzle)

    assert chess_env._puzzle_id == "test_mate_in_1"
    assert "White" in obs  # It's White's turn
    assert not chess_env._is_solved


@pytest.mark.asyncio
async def test_make_correct_move(chess_env):
    """Test making the correct puzzle move."""
    puzzle = {
        "puzzle_id": "test_mate_in_1",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "moves": "h5f7"
    }

    await chess_env.reset(puzzle)

    # Make the correct move (Qxf7#)
    result = await chess_env.step("h5f7")

    assert "Correct" in result or "solved" in result.lower()
    assert chess_env._is_solved


@pytest.mark.asyncio
async def test_make_move_san_format(chess_env):
    """Test making a move in SAN format."""
    puzzle = {
        "puzzle_id": "test_mate_in_1",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "moves": "h5f7"
    }

    await chess_env.reset(puzzle)

    # Make the correct move in SAN format
    result = await chess_env.step("Qxf7#")

    assert "Correct" in result or "solved" in result.lower()
    assert chess_env._is_solved


@pytest.mark.asyncio
async def test_make_illegal_move(chess_env):
    """Test that illegal moves are rejected."""
    await chess_env.reset()

    # Try an illegal move
    result = await chess_env.step("a1a8")  # Can't move like this

    assert "Illegal" in result or "Invalid" in result


@pytest.mark.asyncio
async def test_get_state(chess_env):
    """Test the get_state action."""
    await chess_env.reset()

    result = await chess_env.step("get_state")

    assert "FEN:" in result
    assert "Turn:" in result


@pytest.mark.asyncio
async def test_get_legal_moves(chess_env):
    """Test the get_legal_moves action."""
    await chess_env.reset()

    result = await chess_env.step("get_legal_moves")

    # Should contain some legal moves
    assert len(result) > 0
    # Should have UCI format moves
    assert "(" in result  # Format is "uci (san)"


@pytest.mark.asyncio
async def test_get_evaluation(chess_env):
    """Test the get_reward/evaluation action."""
    await chess_env.reset()

    result = await chess_env.step("get_reward")

    assert isinstance(result, dict)
    assert "reward" in result


@pytest.mark.asyncio
async def test_get_best_move(chess_env):
    """Test getting the best move from Stockfish."""
    puzzle = {
        "puzzle_id": "test",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "moves": "h5f7"
    }

    await chess_env.reset(puzzle)

    best_move, cp = await chess_env.get_best_move()

    # Stockfish should find the mate
    assert best_move == "h5f7"  # Qxf7#


@pytest.mark.asyncio
async def test_puzzle_state_tracking(chess_env):
    """Test that puzzle state is tracked correctly."""
    puzzle = {
        "puzzle_id": "test",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves": "e2e4 e7e5 g1f3"  # Multi-move puzzle: e2e4 is setup, e7e5 is agent's move, g1f3 is response
    }

    await chess_env.reset(puzzle)

    # Initial state
    assert len(chess_env.moves_made) == 0
    assert not chess_env.is_solved

    # After setup, solution index should be 1 (first move was played)
    assert chess_env._current_solution_idx == 1

    # Make the correct move (e7e5, not e2e4 which was already auto-played as setup)
    await chess_env.step("e7e5")

    assert len(chess_env.moves_made) == 1
    assert "e7e5" in chess_env.moves_made


@pytest.mark.asyncio
async def test_multiple_puzzle_resets(chess_env):
    """Test that environment can be reset multiple times."""
    for i in range(3):
        obs = await chess_env.reset({
            "puzzle_id": f"test_{i}",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": "e2e4"
        })

        assert chess_env._puzzle_id == f"test_{i}"
        assert len(chess_env.moves_made) == 0
        assert not chess_env.is_solved


@pytest.mark.asyncio
async def test_board_property(chess_env):
    """Test that board property returns the current board."""
    await chess_env.reset()

    board = chess_env.board

    assert board is not None
    # Initial position should have all pieces
    assert len(list(board.legal_moves)) > 0
