#!/usr/bin/env python3
"""Build AgentFly-format ALFWorld task files from the official ALFWorld data.

Downloads the official data straight from ALFWorld's public GitHub releases (no
`alfworld` package needed) and emits one JSON task file per split — the same games
verl-agent uses (full ``train`` for training, ``valid_seen`` / ``valid_unseen`` for
eval), as explicit task-id lists.

Two official release zips are merged (both extract under ``json_2.1.1/``):
  * json_2.1.1_json.zip   -> traj_data.json  (task id, type, description)
  * json_2.1.2_tw-pddl.zip -> game.tw-pddl   (marks which trials are *playable*)
Only trials that have a game.tw-pddl are emitted, so every task_id is resettable —
this yields the canonical 3553 train / 140 valid_seen / 134 valid_unseen games.

Each task dict matches what the rollout/env expect:

    {
      "question": "Complete this household task: <goal>",
      "answer": "I will complete this household task step by step ...",
      "messages": [{"role": "user", "content": "Complete this household task: <goal>"}],
      "task_id": "trial_T2019...",          # the trial FOLDER name (what env.reset loads)
      "split": "train" | "valid_seen" | "valid_unseen",
      "task_description": "<goal>",
      "task_type": "pick_and_place_simple",
      "data_source": "alfworld_<split>.json"
    }

Usage:
    python prepare_alfworld.py --out-dir data/rlhf/alfworld
    python prepare_alfworld.py --splits train valid_unseen
    python prepare_alfworld.py --alfworld-data ~/.cache/alfworld   # reuse an existing extract
"""
import argparse
import glob
import json
import os
import sys
import urllib.request
import zipfile
from collections import Counter

# Official ALFWorld data (public GitHub release assets, no auth required).
JSON_URL = "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip"
TW_PDDL_URL = "https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip"

ANSWER = "I will complete this household task step by step using the available actions in the AlfWorld environment."
SPLITS = ["train", "valid_seen", "valid_unseen"]
OUT_NAME = {
    "train": "alfworld_train_tasks.json",
    "valid_seen": "alfworld_valid_seen_tasks.json",
    "valid_unseen": "alfworld_valid_unseen_tasks.json",
}


def default_alfworld_data() -> str:
    return os.path.expanduser(os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld"))


def _download_and_extract(url: str, dest_dir: str) -> None:
    """Stream a release zip to a temp file and extract it under dest_dir."""
    tmp = os.path.join(dest_dir, "_download.zip")
    print(f"[alfworld] downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r[alfworld]   {read / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(dest_dir)
    os.remove(tmp)


def ensure_data(alfworld_data: str) -> str:
    """Return the json_2.1.1 root, downloading + merging the official zips if needed."""
    root = os.path.join(alfworld_data, "json_2.1.1")
    have_games = glob.glob(os.path.join(root, "train", "*", "*", "game.tw-pddl"))
    if have_games:
        return root
    os.makedirs(alfworld_data, exist_ok=True)
    _download_and_extract(JSON_URL, alfworld_data)      # traj_data.json tree
    _download_and_extract(TW_PDDL_URL, alfworld_data)   # game.tw-pddl (merges in place)
    if not glob.glob(os.path.join(root, "train", "*", "*", "game.tw-pddl")):
        raise SystemExit(f"Extraction finished but no game.tw-pddl found under {root}.")
    return root


def task_description(traj: dict) -> str:
    """First human (turk) annotation, matching AgentFly's existing task files.

    The env renders its own goal at reset, so this only frames the first user turn.
    """
    for ann in (traj.get("turk_annotations") or {}).get("anns") or []:
        desc = (ann.get("task_desc") or "").strip()
        if desc:
            return desc[0].upper() + desc[1:]
    return (traj.get("task_type") or "complete the task").replace("_", " ")


def build_split(root: str, split: str) -> list:
    """One task dict per *playable* trial (has game.tw-pddl) under json_2.1.1/<split>/."""
    pattern = os.path.join(root, split, "*", "*", "game.tw-pddl")
    tasks, seen = [], set()
    for game_path in sorted(glob.glob(pattern)):
        trial_dir = os.path.dirname(game_path)
        task_id = os.path.basename(trial_dir)              # trial_T2019..., the env's key
        if task_id in seen:
            continue
        traj_path = os.path.join(trial_dir, "traj_data.json")
        if not os.path.exists(traj_path):
            continue
        seen.add(task_id)
        task_type = os.path.basename(os.path.dirname(trial_dir)).split("-")[0]
        try:
            with open(traj_path) as fh:
                traj = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] skipping {traj_path}: {exc}", file=sys.stderr)
            continue
        desc = task_description(traj)
        question = f"Complete this household task: {desc}"
        tasks.append({
            "question": question,
            "answer": ANSWER,
            "messages": [{"role": "user", "content": question}],
            "task_id": task_id,
            "split": split,
            "task_description": desc,
            "task_type": task_type,
            "data_source": f"alfworld_{split}.json",
        })
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alfworld-data", default=default_alfworld_data(),
                    help="Parent of json_2.1.1 to download into / reuse (default: $ALFWORLD_DATA or ~/.cache/alfworld)")
    ap.add_argument("--out-dir", default="data/rlhf/alfworld", help="Where to write the task files")
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS,
                    help="Which official splits to build (default: all three)")
    args = ap.parse_args()

    root = ensure_data(args.alfworld_data)
    os.makedirs(args.out_dir, exist_ok=True)

    for split in args.splits:
        tasks = build_split(root, split)
        out_path = os.path.join(args.out_dir, OUT_NAME[split])
        with open(out_path, "w") as fh:
            json.dump(tasks, fh, indent=2)
        dist = dict(Counter(t["task_type"] for t in tasks))
        print(f"[alfworld] {split}: {len(tasks)} tasks -> {out_path}")
        print(f"           task_type dist: {dist}")


if __name__ == "__main__":
    main()
