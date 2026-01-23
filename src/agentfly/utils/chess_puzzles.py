# chess_puzzles.py
"""
Chess puzzle data loading utilities for AgentFly.

Supports loading puzzles from:
- Lichess puzzle database CSV format
- JSONL format for training

The Lichess puzzle database can be downloaded from:
https://database.lichess.org/#puzzles
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union


def load_lichess_puzzles(
    csv_path: Union[str, Path],
    max_puzzles: Optional[int] = None,
    min_rating: int = 0,
    max_rating: int = 3000,
    themes: Optional[List[str]] = None,
    skip_first_n: int = 0,
) -> List[Dict]:
    """
    Load puzzles from Lichess puzzle database CSV.

    The Lichess puzzle CSV format has columns:
    PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

    Args:
        csv_path: Path to the Lichess puzzles CSV file
        max_puzzles: Maximum number of puzzles to load (None for all)
        min_rating: Minimum puzzle rating to include
        max_rating: Maximum puzzle rating to include
        themes: Optional list of themes to filter by (e.g., ["mateIn1", "short"])
                If provided, only puzzles with at least one matching theme are included
        skip_first_n: Number of puzzles to skip from the beginning

    Returns:
        List of puzzle dictionaries in AgentFly format with keys:
            - messages: List of message dicts for the conversation
            - puzzle_id: Unique puzzle identifier
            - fen: Starting FEN position
            - moves: Space-separated UCI moves (solution)
            - rating: Puzzle difficulty rating
            - themes: List of puzzle themes

    Example:
        ```python
        puzzles = load_lichess_puzzles(
            "lichess_puzzles.csv",
            max_puzzles=1000,
            min_rating=1200,
            max_rating=1800,
            themes=["mateIn2"]
        )
        ```
    """
    puzzles = []
    csv_path = Path(csv_path)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        skipped = 0
        for row in reader:
            # Skip first N puzzles
            if skipped < skip_first_n:
                skipped += 1
                continue

            # Filter by rating
            try:
                rating = int(row["Rating"])
            except (ValueError, KeyError):
                continue

            if rating < min_rating or rating > max_rating:
                continue

            # Filter by themes
            puzzle_themes = row.get("Themes", "").split()
            if themes and not any(t in puzzle_themes for t in themes):
                continue

            # Create puzzle in AgentFly format
            puzzle = {
                "messages": [
                    {
                        "role": "user",
                        "content": generate_puzzle_prompt(
                            rating=rating, themes=puzzle_themes
                        ),
                    }
                ],
                "puzzle_id": row["PuzzleId"],
                "fen": row["FEN"],
                "moves": row["Moves"],
                "rating": rating,
                "themes": puzzle_themes,
            }
            puzzles.append(puzzle)

            if max_puzzles and len(puzzles) >= max_puzzles:
                break

    return puzzles


def load_puzzles_jsonl(
    jsonl_path: Union[str, Path],
    max_puzzles: Optional[int] = None,
) -> List[Dict]:
    """
    Load puzzles from JSONL file.

    Each line should be a JSON object with at least:
    - fen: Starting FEN position
    - moves: Space-separated UCI moves (solution)

    Optional fields:
    - puzzle_id: Unique identifier
    - rating: Difficulty rating
    - themes: List of themes
    - messages: Pre-formatted messages

    Args:
        jsonl_path: Path to the JSONL file
        max_puzzles: Maximum number of puzzles to load

    Returns:
        List of puzzle dictionaries
    """
    puzzles = []
    jsonl_path = Path(jsonl_path)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_puzzles and len(puzzles) >= max_puzzles:
                break

            line = line.strip()
            if not line:
                continue

            try:
                puzzle = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Ensure required fields
            if "fen" not in puzzle or "moves" not in puzzle:
                continue

            # Add default fields if missing
            if "puzzle_id" not in puzzle:
                puzzle["puzzle_id"] = f"puzzle_{i}"

            if "messages" not in puzzle:
                puzzle["messages"] = [
                    {
                        "role": "user",
                        "content": generate_puzzle_prompt(
                            rating=puzzle.get("rating"), themes=puzzle.get("themes", [])
                        ),
                    }
                ]

            puzzles.append(puzzle)

    return puzzles


def generate_puzzle_prompt(
    rating: Optional[int] = None,
    themes: Optional[List[str]] = None,
    include_hints: bool = True,
) -> str:
    """
    Generate the user prompt for a chess puzzle.

    Args:
        rating: Puzzle difficulty rating
        themes: List of puzzle themes
        include_hints: Whether to include hints based on themes

    Returns:
        Formatted prompt string for the agent
    """
    parts = ["You are solving a chess puzzle."]

    if rating:
        parts.append(f"Difficulty rating: {rating}")

    if include_hints and themes:
        # Add hints based on common themes
        if "mateIn1" in themes:
            parts.append("Hint: This is a mate in 1 - find the checkmate!")
        elif "mateIn2" in themes:
            parts.append("Hint: This is a mate in 2 moves.")
        elif "mateIn3" in themes:
            parts.append("Hint: This is a mate in 3 moves.")
        elif "mateIn4" in themes:
            parts.append("Hint: This is a mate in 4 moves.")
        elif "fork" in themes:
            parts.append("Hint: Look for a fork (attacking multiple pieces at once).")
        elif "pin" in themes:
            parts.append("Hint: Look for a pin.")
        elif "skewer" in themes:
            parts.append("Hint: Look for a skewer.")
        elif "discoveredAttack" in themes:
            parts.append("Hint: Look for a discovered attack.")

    parts.extend(
        [
            "",
            "Use the chess_get_state tool to see the current board position.",
            "Use the chess_get_legal_moves tool to see available moves.",
            "Use the chess_move tool to make your move(s).",
            "",
            "Find the best move(s) to solve this puzzle.",
        ]
    )

    return "\n".join(parts)


def filter_puzzles_by_theme(
    puzzles: List[Dict], themes: List[str], require_all: bool = False
) -> List[Dict]:
    """
    Filter puzzles by themes.

    Args:
        puzzles: List of puzzle dictionaries
        themes: Themes to filter by
        require_all: If True, puzzle must have ALL themes. If False, any matching theme.

    Returns:
        Filtered list of puzzles
    """
    result = []
    for puzzle in puzzles:
        puzzle_themes = puzzle.get("themes", [])
        if require_all:
            if all(t in puzzle_themes for t in themes):
                result.append(puzzle)
        else:
            if any(t in puzzle_themes for t in themes):
                result.append(puzzle)
    return result


def filter_puzzles_by_rating(
    puzzles: List[Dict], min_rating: int = 0, max_rating: int = 3000
) -> List[Dict]:
    """
    Filter puzzles by rating range.

    Args:
        puzzles: List of puzzle dictionaries
        min_rating: Minimum rating (inclusive)
        max_rating: Maximum rating (inclusive)

    Returns:
        Filtered list of puzzles
    """
    return [p for p in puzzles if min_rating <= p.get("rating", 0) <= max_rating]


def save_puzzles_jsonl(
    puzzles: List[Dict],
    output_path: Union[str, Path],
) -> None:
    """
    Save puzzles to JSONL file.

    Args:
        puzzles: List of puzzle dictionaries
        output_path: Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for puzzle in puzzles:
            f.write(json.dumps(puzzle, ensure_ascii=False) + "\n")


# Common Lichess puzzle themes for reference
LICHESS_THEMES = [
    # Tactical motifs
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "doubleCheck",
    "sacrifice",
    "deflection",
    "interference",
    "xRayAttack",
    "zugzwang",
    "quietMove",
    "defensiveMove",
    "clearance",
    # Mate patterns
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "mateIn4",
    "mateIn5",
    "anastasiaMate",
    "arabianMate",
    "backRankMate",
    "bodenMate",
    "doubleBishopMate",
    "hookMate",
    "smotheredMate",
    # Length
    "oneMove",
    "short",
    "long",
    "veryLong",
    # Game phase
    "opening",
    "middlegame",
    "endgame",
    "rookEndgame",
    "bishopEndgame",
    "knightEndgame",
    "pawnEndgame",
    "queenEndgame",
    # Difficulty
    "master",
    "masterVsMaster",
    "superGM",
    # Special
    "castling",
    "enPassant",
    "promotion",
    "underPromotion",
    "equality",
    "advantage",
    "crushing",
]
