"""Centralized, approval-gated data downloads for AgentFly.

All bulk data — training/eval task files and tool corpora/indices — is fetched
through this module so a user always sees *what* will be downloaded and *how big*
it is before anything hits the network. This is especially important for new users
who run a training script without the data present locally.

Two entry points:
  * ``ensure_training_data(config)`` — called by the trainer preflight; resolves the
    ``data.train_files`` / ``data.val_files`` paths against the ``Agent-One/AgentFly-Train``
    HF dataset and downloads any that are missing.
  * ``download_tool_data(tool_name)`` — fetches large tool assets (e.g. the search
    corpus + FAISS index) on demand.

Approval:
  * Interactive TTY  -> a ``[y/N]`` prompt showing per-file and total size.
  * ``AF_ASSUME_YES=1`` -> proceed without prompting (CI / batch jobs).
  * Non-interactive without that env var -> treated as a decline, so batch jobs
    never silently pull tens of GB.
"""

import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url

from .. import AF_CACHE_DIR

# Canonical HF dataset holding AgentFly training/eval task files.
TRAIN_REPO = "Agent-One/AgentFly-Train"

# Script-referenced basenames whose file lives under a different name in TRAIN_REPO.
TRAIN_ALIASES: Dict[str, str] = {}

_train_repo_cache: Optional[Dict[str, int]] = None


# --------------------------------------------------------------------------- #
# Formatting / approval
# --------------------------------------------------------------------------- #


def human_size(num_bytes: Optional[int]) -> str:
    if not num_bytes:
        return "unknown size"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _assume_yes() -> bool:
    return os.environ.get("AF_ASSUME_YES", "").strip().lower() in ("1", "true", "yes", "y")


def confirm(plan: Sequence[Tuple[str, Optional[int]]], *, what: str) -> bool:
    """Show a size-annotated download plan and ask for approval.

    ``plan`` is a list of ``(label, size_bytes)``. Returns True if the download
    should proceed. Honors ``AF_ASSUME_YES``; a non-TTY without it declines.
    """
    total = sum(s or 0 for _, s in plan)
    print(f"\n[AgentFly] Need to download {what} ({human_size(total)} total):", file=sys.stderr)
    for label, size in plan:
        print(f"    {label}  ({human_size(size)})", file=sys.stderr)

    if _assume_yes():
        print("[AgentFly] AF_ASSUME_YES set -> proceeding.\n", file=sys.stderr)
        return True
    if not sys.stdin or not sys.stdin.isatty():
        print(
            "[AgentFly] Non-interactive shell: set AF_ASSUME_YES=1 to allow this "
            "download (or fetch the files manually). Skipping.\n",
            file=sys.stderr,
        )
        return False
    try:
        answer = input("[AgentFly] Proceed with download? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _hf_file_size(repo_id: str, filename: str, repo_type: str = "dataset") -> int:
    try:
        meta = get_hf_file_metadata(
            hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)
        )
        return int(meta.size or 0)
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Training / eval task files (Agent-One/AgentFly-Train)
# --------------------------------------------------------------------------- #


def _train_repo_files() -> Dict[str, int]:
    """Map of ``filename -> size_bytes`` for TRAIN_REPO (fetched once, cached)."""
    global _train_repo_cache
    if _train_repo_cache is None:
        info = HfApi().dataset_info(TRAIN_REPO, files_metadata=True)
        _train_repo_cache = {
            s.rfilename: int(s.size or 0)
            for s in info.siblings
            if s.rfilename.endswith(".json")
        }
    return _train_repo_cache


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def ensure_training_files(local_paths: Sequence[str]) -> None:
    """Ensure each local task-file path exists, downloading missing ones from
    TRAIN_REPO by basename (via TRAIN_ALIASES when the repo name differs).

    Files not present in TRAIN_REPO are skipped with a note — they must be
    provided by the user (e.g. R2E-Gym, GUI, infoseek datasets).
    """
    repo_files = None
    plan: List[Tuple[str, str, int]] = []  # (local_path, hf_name, size)
    for path in local_paths:
        if not path or os.path.exists(path):
            continue
        base = os.path.basename(path)
        hf_name = TRAIN_ALIASES.get(base, base)
        if repo_files is None:
            repo_files = _train_repo_files()
        if hf_name in repo_files:
            plan.append((path, hf_name, repo_files[hf_name]))
        else:
            print(
                f"[AgentFly] '{base}' is not in {TRAIN_REPO}; provide it manually at {path}.",
                file=sys.stderr,
            )

    if not plan:
        return

    labels = [(f"{hf}  ->  {lp}", size) for lp, hf, size in plan]
    if not confirm(labels, what=f"{len(plan)} training file(s) from {TRAIN_REPO}"):
        raise RuntimeError(
            "Training data download declined. Re-run interactively and accept, set "
            "AF_ASSUME_YES=1, or place the files manually at the paths above."
        )

    for local_path, hf_name, _ in plan:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        cached = hf_hub_download(repo_id=TRAIN_REPO, filename=hf_name, repo_type="dataset")
        shutil.copyfile(cached, local_path)
        print(f"[AgentFly] downloaded {hf_name} -> {local_path}", file=sys.stderr)


def ensure_training_data(config) -> None:
    """Trainer preflight: resolve+download the configured train/val task files."""
    data_cfg = getattr(config, "data", None)
    if data_cfg is None:
        return
    paths = _as_list(data_cfg.get("train_files")) + _as_list(data_cfg.get("val_files"))
    ensure_training_files(paths)


# --------------------------------------------------------------------------- #
# Tool assets (search corpus + FAISS index, etc.)
# --------------------------------------------------------------------------- #


def _ensure_hf_files(
    repo_id: str,
    filenames: Sequence[str],
    dest_dir: str,
    *,
    repo_type: str = "dataset",
    what: str,
) -> None:
    """Download any of ``filenames`` from ``repo_id`` missing under ``dest_dir``,
    after a single approval prompt."""
    os.makedirs(dest_dir, exist_ok=True)
    missing = [f for f in filenames if not os.path.exists(os.path.join(dest_dir, f))]
    if not missing:
        return
    plan = [(f"{repo_id}/{f}", _hf_file_size(repo_id, f, repo_type)) for f in missing]
    if not confirm(plan, what=what):
        raise RuntimeError(
            f"Download of {what} declined. Set AF_ASSUME_YES=1 to allow non-interactively."
        )
    for f in missing:
        hf_hub_download(repo_id=repo_id, filename=f, repo_type=repo_type, local_dir=dest_dir)


def download_tool_data(tool_name: str) -> None:
    """Download large assets a tool needs, gated by an approval prompt."""
    if tool_name == "asyncdense_retrieve":
        data_dir = os.path.join(AF_CACHE_DIR, "data", "search")
        corpus_file = os.path.join(data_dir, "wiki-18.jsonl")
        index_file = os.path.join(data_dir, "e5_Flat.index")

        if not os.path.exists(corpus_file) and not os.path.exists(
            os.path.join(data_dir, "wiki-18.jsonl.gz")
        ):
            _ensure_hf_files(
                "PeterJinGo/wiki-18-corpus",
                ["wiki-18.jsonl.gz"],
                data_dir,
                what="search corpus (wiki-18)",
            )
        gz_path = os.path.join(data_dir, "wiki-18.jsonl.gz")
        if not os.path.exists(corpus_file) and os.path.exists(gz_path):
            import gzip

            print(f"[AgentFly] unzipping {gz_path}", file=sys.stderr)
            with gzip.open(gz_path, "rb") as f_in, open(corpus_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        if not os.path.exists(index_file):
            if not os.path.exists(os.path.join(data_dir, "part_aa")):
                _ensure_hf_files(
                    "PeterJinGo/wiki-18-e5-index",
                    ["part_aa", "part_ab"],
                    data_dir,
                    what="search FAISS index (e5_Flat)",
                )
            print(f"[AgentFly] concatenating index parts -> {index_file}", file=sys.stderr)
            os.system(f"cat {os.path.join(data_dir, 'part_*')} > {index_file}")
    else:
        raise ValueError(f"Unknown tool data set: {tool_name!r}")


if __name__ == "__main__":
    download_tool_data("asyncdense_retrieve")
