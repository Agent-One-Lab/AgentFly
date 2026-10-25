"""Write the number-guessing task files: ``python examples/custom_env/make_data.py``."""

import json
import os
import random

PROMPT = (
    "I'm thinking of a whole number between 1 and 100. Find it with the guess tool, "
    "which tells you whether the number is higher or lower than your guess."
)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(out_dir, exist_ok=True)
rng = random.Random(0)
for split, n in (("train", 1000), ("val", 100)):
    # "question" becomes the user message; "target" is kept as metadata for the env.
    tasks = [{"question": PROMPT, "target": rng.randint(1, 100)} for _ in range(n)]
    path = os.path.join(out_dir, f"guess_{split}.json")
    with open(path, "w") as f:
        json.dump(tasks, f, indent=1)
    print(f"wrote {n} tasks to {path}")
