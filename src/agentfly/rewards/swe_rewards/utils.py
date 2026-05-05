from __future__ import annotations

import asyncio
from typing import Any


def remove_binary_diffs(patch_text):
    """
    Remove binary file diffs from a git patch.

    Args:
        patch_text (str): The git patch text

    Returns:
        str: The cleaned patch text with binary diffs removed
    """
    lines = patch_text.splitlines()
    cleaned_lines = []
    block = []
    is_binary_block = False

    for line in lines:
        if line.startswith("diff --git "):
            if block and not is_binary_block:
                cleaned_lines.extend(block)
            block = [line]
            is_binary_block = False
        elif "Binary files" in line:
            is_binary_block = True
            block.append(line)
        else:
            block.append(line)

    if block and not is_binary_block:
        cleaned_lines.extend(block)
    return "\n".join(cleaned_lines)


# avoid depend on file cmd
def remove_binary_files_from_git():
    """
    Generate a bash command to remove binary files from git staging.

    Returns:
        str: A bash command that removes binary files from git staging
    """
    return """
    for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
        if [ -f "$file" ] && (test -x "$file" || git check-attr binary "$file" | grep -q "binary: set"); then
            git rm -f "$file" 2>/dev/null || rm -f "$file"
            echo "Removed: $file"
        fi
    done
    """.strip()


async def get_patch_from_runtime(
    container: Any,
    *,
    instance: Any = None,
    dataset: str = "",
    workspace_dir_name: str | None = None,
    timeout: int = 300,
) -> str | None:
    """
    Obtain the patch string (git diff) from an Enroot-based container where the agent has run.

    The container is assumed to be running with the agent's edits on disk. This function
    runs a single compound shell command that stages changes, writes the diff to patch.diff,
    then cats it; the output is the patch text.

    Args:
        container: Enroot container resource with async run_cmd(cmd: str) returning str (or bytes).
            Typically agentfly.resources.ContainerResource.
        instance: Optional instance dict; if provided and dataset is not swe-smith/r2e-gym,
            base_commit from instance is used for `git diff --cached <base_commit>`.
        dataset: Optional dataset name; if "swe-smith" or "r2e-gym", uses
            `git diff --no-color --cached` without a base commit.
        workspace_dir_name: Optional path to repo root inside the container (e.g. "/testbed").
            If provided, commands are run from this directory.
        timeout: Reserved for future use (Enroot/exec_run timeout is typically set at container start).

    Returns:
        The patch string (unified diff), or None if obtaining the patch failed.
    """
    work_dir = workspace_dir_name or "."
    if dataset and ("swe-smith" in dataset or "r2e-gym" in dataset):
        diff_cmd = "git diff --no-color --cached > patch.diff"
    else:
        base = (instance or {}).get("base_commit", "HEAD")
        diff_cmd = f"git diff --no-color --cached {base} > patch.diff"

    remove_bin_cmd = remove_binary_files_from_git()
    compound = (
        f"cd {work_dir!r} && "
        'git config --global core.pager "" && '
        "git add -A && "
        f"({remove_bin_cmd}) && "
        f"{diff_cmd} && "
        "cat patch.diff"
    )

    raw = await container.run_cmd(compound, timeout=timeout)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        git_patch = raw.decode("utf-8", errors="replace")
    else:
        git_patch = raw

    if not (git_patch and git_patch.strip()):
        return None

    # Run CPU-bound cleanup in thread pool to avoid blocking the event loop
    return await asyncio.to_thread(remove_binary_diffs, git_patch)
