# Chess Puzzle Solving Agent

Train an RL agent to solve chess puzzles using AgentFly.

## Overview

This module provides a complete chess puzzle solving environment for training language model agents with reinforcement learning. The agent learns to analyze chess positions and find tactical solutions (checkmates, forks, pins, etc.).

**Architecture:**
```
Agent (Qwen/Llama/etc.)
    ↓
Tools: chess_move, chess_get_state, chess_get_legal_moves
    ↓
ChessPuzzleEnv (python-chess + Stockfish)
    ↓
Rewards: chess_puzzle_reward (dense) or chess_puzzle_reward_simple (binary)
```

## Prerequisites

1. **Stockfish chess engine:**
   ```bash
   # macOS
   brew install stockfish

   # Ubuntu/Debian
   apt-get install stockfish

   # Verify installation
   which stockfish
   ```

2. **Python dependencies:**
   ```bash
   pip install python-chess
   ```

## Quick Start

### Option 1: Test Run with Sample Data

The repo includes sample puzzles for testing:

```bash
bash scripts/train_chess.sh
```

### Option 2: Train with Lichess Puzzles

**Step 1: Download Lichess puzzle database**
```bash
# Download (~250MB compressed, ~1.5GB uncompressed)
curl -O https://database.lichess.org/lichess_db_puzzle.csv.zst

# Decompress (install zstd if needed: brew install zstd)
zstd -d lichess_db_puzzle.csv.zst
```

**Step 2: Prepare training data**
```bash
python scripts/prepare_chess_data.py \
    --input lichess_db_puzzle.csv \
    --output data/chess/ \
    --train-size 10000 \
    --val-size 1000 \
    --min-rating 1000 \
    --max-rating 1600 \
    --themes mateIn1 mateIn2 fork pin
```

**Step 3: Run training**
```bash
# Set your WandB key for logging
export WANDB_API_KEY="your_key_here"

# Start training
bash scripts/train_chess.sh
```

## Configuration

Edit `scripts/train_chess.sh` to customize:

### Model
```bash
model="Qwen/Qwen2.5-3B-Instruct"  # Base model to fine-tune
template="qwen2.5"                 # Chat template

# Alternatives:
# model="Qwen/Qwen2.5-7B-Instruct"
# model="meta-llama/Llama-3.1-8B-Instruct"
# template="llama3"
```

### Agent
```bash
agent_type="react"                 # ReAct agent for tool use
max_turns=10                       # Max moves per puzzle
num_chains=8                       # Parallel rollouts per sample
```

### Reward Function
```bash
# Dense reward (recommended) - based on Stockfish evaluation
reward_name="chess_puzzle_reward"

# Binary reward - 1.0 if solved, 0.0 otherwise
# reward_name="chess_puzzle_reward_simple"
```

### Training
```bash
batch_size=64
lr=4e-7
total_training_steps=200
adv_estimator="grpo"  # Options: grpo, reinforce_plus_plus, rloo, gae
```

## Data Format

Training data is a JSON array of puzzles:

```json
[
    {
        "question": "You are solving a chess puzzle...",
        "puzzle_id": "abc123",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "moves": "h5f7",
        "rating": 1200,
        "themes": ["mateIn1", "short"]
    }
]
```

| Field | Description |
|-------|-------------|
| `question` | Prompt shown to the agent |
| `puzzle_id` | Unique identifier |
| `fen` | Starting position (FEN notation) |
| `moves` | Solution moves (space-separated UCI) |
| `rating` | Difficulty rating (optional) |
| `themes` | Puzzle themes (optional) |

### Move Sequence Convention

For multi-move puzzles, the `moves` field follows Lichess convention:
- **First move**: Opponent's setup move (auto-played by environment)
- **Remaining moves**: Alternating solution moves

Example: `"e2e4 e7e5 g1f3"` means:
1. Environment plays `e2e4` (opponent setup)
2. Agent must find `e7e5`
3. Environment responds with `g1f3`

## Available Tools

| Tool | Description |
|------|-------------|
| `chess_move` | Make a move (UCI: `e2e4` or SAN: `Nf3`) |
| `chess_get_state` | View board position, FEN, turn, status |
| `chess_get_legal_moves` | List all legal moves |

## Reward Functions

### `chess_puzzle_reward` (Dense)

Combines solve bonus with move quality:

```
reward = 0.6 × solve_reward + 0.4 × move_quality_reward
```

- `solve_reward`: 1.0 if solved, 0-0.5 for partial progress
- `move_quality_reward`: Average centipawn quality (Stockfish evaluation)

### `chess_puzzle_reward_simple` (Binary)

- 1.0 if puzzle solved correctly
- 0.0 otherwise

## Puzzle Themes

Filter puzzles by tactical motif:

| Category | Themes |
|----------|--------|
| **Mates** | `mateIn1`, `mateIn2`, `mateIn3`, `backRankMate`, `smotheredMate` |
| **Tactics** | `fork`, `pin`, `skewer`, `discoveredAttack`, `doubleCheck` |
| **Length** | `oneMove`, `short`, `long`, `veryLong` |
| **Phase** | `opening`, `middlegame`, `endgame` |

## Monitoring

Training metrics are logged to Weights & Biases:
- Reward per step
- Solve rate
- Average moves per puzzle
- KL divergence
- Loss curves

View at: https://wandb.ai/your-project

## Files

```
AgentFly/
├── agentfly/
│   ├── envs/chess_env.py           # Chess puzzle environment
│   ├── tools/src/chess/tools.py    # Agent tools
│   ├── rewards/chess_reward.py     # Reward functions
│   └── utils/chess_puzzles.py      # Data loading utilities
├── scripts/
│   ├── train_chess.sh              # Training script
│   └── prepare_chess_data.py       # Lichess → training format
└── data/chess/
    ├── chess_puzzles_train.json    # Training data
    └── chess_puzzles_val.json      # Validation data
```

## Troubleshooting

### Stockfish not found
```
FileNotFoundError: [Errno 2] No such file or directory: '/opt/homebrew/bin/stockfish'
```
**Fix:** Update `stockfish_path` in `chess_env.py` or install Stockfish.

### Out of GPU memory
**Fix:** Reduce `batch_size`, enable offloading:
```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

### Slow training
- Reduce `analysis_depth` in `ChessPuzzleEnv` (default: 20)
- Use simpler puzzles (lower rating, `mateIn1` only)
- Decrease `num_chains`


### For real training, use Lichess data:

# Download 4+ million verified puzzles
curl -O https://database.lichess.org/lichess_db_puzzle.csv.zst
zstd -d lichess_db_puzzle.csv.zst

# Convert to training format
python scripts/prepare_chess_data.py \
--input lichess_db_puzzle.csv \
--output data/chess/ \
--train-size 10000 \
--val-size 1000

The Lichess puzzles are:
- Extracted from real games
- Validated by millions of players
- Rated by difficulty (Elo)
- Tagged with themes (mateIn1, fork, pin, etc.)


## References

- [Lichess Puzzle Database](https://database.lichess.org/#puzzles)
- [python-chess Documentation](https://python-chess.readthedocs.io/)
- [Stockfish](https://stockfishchess.org/)
