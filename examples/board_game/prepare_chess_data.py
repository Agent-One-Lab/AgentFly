#!/usr/bin/env python3
"""
Prepare chess puzzle training data from Lichess puzzle database.

Usage:
    python prepare_chess_data.py --input lichess_db_puzzle.csv --output data/chess/

Downloads:
    Get the puzzle database from: https://database.lichess.org/#puzzles

    curl -O https://database.lichess.org/lichess_db_puzzle.csv.zst
    zstd -d lichess_db_puzzle.csv.zst
"""

import argparse
import json
import random
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentfly.utils.chess_puzzles import (
    load_lichess_puzzles,
    save_puzzles_jsonl,
    filter_puzzles_by_rating,
    filter_puzzles_by_theme,
    LICHESS_THEMES,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare chess puzzle training data")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to Lichess puzzle CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/chess/",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=10000,
        help="Number of training puzzles"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1000,
        help="Number of validation puzzles"
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=1000,
        help="Minimum puzzle rating"
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=1800,
        help="Maximum puzzle rating"
    )
    parser.add_argument(
        "--themes",
        type=str,
        nargs="+",
        default=["mateIn1", "mateIn2", "mateIn3", "fork", "pin"],
        help="Puzzle themes to include"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print(f"Loading puzzles from {args.input}...")
    print(f"  Rating range: {args.min_rating} - {args.max_rating}")
    print(f"  Themes: {args.themes}")

    # Load puzzles with filters
    total_needed = args.train_size + args.val_size
    puzzles = load_lichess_puzzles(
        csv_path=args.input,
        max_puzzles=total_needed * 2,  # Load extra for filtering
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        themes=args.themes,
    )

    print(f"Loaded {len(puzzles)} puzzles matching criteria")

    if len(puzzles) < total_needed:
        print(f"Warning: Only found {len(puzzles)} puzzles, need {total_needed}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(puzzles)

    train_puzzles = puzzles[:args.train_size]
    val_puzzles = puzzles[args.train_size:args.train_size + args.val_size]

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON (AgentFly format)
    train_path = output_dir / "chess_puzzles_train.json"
    val_path = output_dir / "chess_puzzles_val.json"

    with open(train_path, "w") as f:
        json.dump(train_puzzles, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val_puzzles, f, indent=2)

    print(f"\nSaved {len(train_puzzles)} training puzzles to {train_path}")
    print(f"Saved {len(val_puzzles)} validation puzzles to {val_path}")

    # Print rating distribution
    train_ratings = [p["rating"] for p in train_puzzles]
    if train_ratings:
        print(f"\nTraining set rating distribution:")
        print(f"  Min: {min(train_ratings)}")
        print(f"  Max: {max(train_ratings)}")
        print(f"  Avg: {sum(train_ratings) / len(train_ratings):.0f}")

    # Print theme distribution
    theme_counts = {}
    for p in train_puzzles:
        for theme in p.get("themes", []):
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

    print(f"\nTop themes in training set:")
    for theme, count in sorted(theme_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {theme}: {count}")


if __name__ == "__main__":
    main()
