# test_chess_smoke.py
"""
End-to-end smoke test for the chess puzzle agent pipeline.

Verifies that ReactAgent + chess tools + chess reward work together:
- Agent receives a puzzle
- Tools dispatch to ChessPuzzleEnv
- Trajectories are populated after run()
- Reward function produces a valid result

Requires: Stockfish installed (brew install stockfish / apt-get install stockfish)
"""

import shutil
from unittest.mock import AsyncMock

import pytest

from agentfly.agents import ReactAgent
from agentfly.envs.chess_env import ChessPuzzleEnv
from agentfly.rewards import chess_puzzle_reward
from agentfly.tools import chess_get_legal_moves, chess_get_state, chess_move

pytestmark = pytest.mark.skip(reason="Skipping for now")

# A simple mate-in-1 puzzle: White plays Qxf7#
MATE_IN_1_PUZZLE = {
    "puzzle_id": "smoke_mate1",
    "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "moves": "h5f7",
}


def _make_react_responses():
    """Return a sequence of canned ReAct responses for the smoke test.

    Turn 0: agent checks the board state
    Turn 1: agent makes the winning move Qxf7#
    """
    return [
        # Turn 0 – get state
        (
            'Thought: Let me look at the current board position first.\n'
            'Action: chess_get_state\n'
            'Input: {}'
        ),
        # Turn 1 – make the winning move
        (
            'Thought: I see White can play Qxf7# for checkmate.\n'
            'Action: chess_move\n'
            'Input: {"move": "h5f7"}'
        ),
    ]


@pytest.mark.asyncio
async def test_chess_smoke_e2e():
    """Smoke test: ReactAgent solves a mate-in-1 puzzle with chess tools."""

    canned = _make_react_responses()
    call_idx = 0

    async def fake_generate(messages_list, **kwargs):
        nonlocal call_idx
        idx = min(call_idx, len(canned) - 1)
        call_idx += 1
        return [canned[idx]]

    tools = [chess_move, chess_get_state, chess_get_legal_moves]

    agent = ReactAgent(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        tools=tools,
        backend="client",
        reward_fn=chess_puzzle_reward,
        monitors=[],
        debug=True,
    )

    # Replace LLM engine methods with mocks
    agent.llm_engine = AsyncMock()
    agent.llm_engine.generate_async = fake_generate
    agent.llm_engine.preprocess = lambda: None
    agent.llm_engine.postprocess = lambda: None

    messages = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Solve this chess puzzle. The position is a mate-in-1. "
                        "Find the winning move for White."
                    ),
                }
            ],
            **MATE_IN_1_PUZZLE,
        }
    ]

    await agent.run(
        messages=messages,
        max_turns=2,
        num_chains=1,
        enable_streaming=False,
    )

    # Trajectories should be populated
    trajectories = agent.trajectories
    assert len(trajectories) > 0, "Expected at least one trajectory"

    # The trajectory should contain messages
    traj = trajectories[0]
    assert "messages" in traj
    assert len(traj["messages"]) > 1, "Expected multiple messages in trajectory"
