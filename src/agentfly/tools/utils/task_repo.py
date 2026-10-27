"""Put a SWE task's repository at the task's commit before an agent's first command.

SWE-smith images start on ``main``, the original code; each task's bug lives on a git branch
named after the task (the dataset's ``git_commit_hash``, which also accepts a commit hash).
:func:`checkout_task_commit` checks it out once per rollout, whichever container-backed SWE
tool (shell, file tools, ``bash``) runs first. Tasks without ``git_commit_hash`` (R2E-Gym,
SWE-bench: their images already hold the task's code) are left alone.
"""

import shlex
from typing import Optional

from ...core import Context

DEFAULT_REPO_DIR = "/testbed"  # where SWE-bench / SWE-smith / R2E-Gym images keep the repository
_DONE_KEY = "_task_commit_checked_out"
_RC_MARKER = "__AGENTFLY_GIT_RC__="
GIT_TIMEOUT = 300


async def _git(container, args: str, repo_dir: str) -> tuple:
    """Run ``git <args>`` in ``repo_dir``; return (exit code, output)."""
    cmd = f"bash -c {shlex.quote(f'git {args}')} 2>&1; printf '\\n{_RC_MARKER}%d' $?"
    raw = await container.run_cmd(cmd, timeout=GIT_TIMEOUT, workdir=repo_dir)
    out = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
    body, _, rc = out.rpartition(_RC_MARKER)
    return (int(rc.strip()) if rc.strip().isdigit() else 1), body.strip()


async def checkout_task_commit(container, context: Context) -> Optional[str]:
    """Check out ``context.metadata['git_commit_hash']`` in the task repo, once per rollout.

    The branch is normally already in the image, so this needs no network; ``git fetch`` runs
    only when the first checkout fails. Returns ``None`` on success (or when there is nothing
    to do), otherwise an error message for the tool to return, so the agent never silently
    works on the wrong code.
    """
    commit = context.metadata.get("git_commit_hash")
    if not commit or context.metadata.get(_DONE_KEY):
        return None
    repo_dir = context.metadata.get("workdir") or DEFAULT_REPO_DIR
    rc, out = await _git(container, f"checkout -q {shlex.quote(commit)}", repo_dir)
    if rc != 0:
        await _git(container, "fetch -q", repo_dir)
        rc, out = await _git(container, f"checkout -q {shlex.quote(commit)}", repo_dir)
    if rc != 0:
        return f"Error: could not check out the task's commit {commit!r} in {repo_dir}: {out[-500:]}"
    context.metadata[_DONE_KEY] = True
    return None
