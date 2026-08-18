"""Three-tool surface for the skill subsystem (small-model variant).

Implements ``load_skill``, ``read_skill_file``, and ``run_skill_script`` as
described in ``skills/skill-infra-spec.md`` §2.3. Each tool returns a
JSON-encoded string so the model gets the structured fields the spec calls
for. Errors are reported as ``{"error": "..."}`` rather than raised, so a
failed tool call still produces a useful observation in the rollout.
"""

import asyncio
import base64
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .... import AF_CONTAINER_SKILLS_DIR
from ....core import Context
from ....resources import ContainerResourceSpec
from ...decorator import tool
from .skill_loader import (
    build_manifest,
    find_skills_root,
    loaded_skills,
    parse_skill_md,
    resolve_manifest_path,
    safe_skill_dir,
)


# read_skill_file output cap. Bounded so a reasoning builder can't blow up its
# own context by reading several large skill files — at ~30k+ token context
# GLM/M3 overrun their per-turn token budget mid-<think> and return an empty turn
# (no tool call -> invalid_no_validate). 16k chars/file truncates only ~10% of
# reads (p90 of observed reads was ~16k) while clipping the outliers that drive
# the context blow-up; paired with --gen-max-tokens 32000.
_MAX_FILE_BYTES = 16_000
_MAX_STREAM_BYTES = 32 * 1024
_MAX_TIMEOUT = 300
_MIN_TIMEOUT = 1

_LOAD_SKILL_DESC = (
    "Load a skill into context. Returns the skill's instructions (SKILL.md body) and "
    "a manifest of files available in the skill. After loading, use read_skill_file "
    "to read reference docs and run_skill_script to execute bundled scripts."
)
_READ_SKILL_FILE_DESC = (
    "Read a file from a loaded skill. Use for reference documentation listed in the "
    "skill's file manifest. The skill must be loaded first via load_skill."
)
_RUN_SKILL_SCRIPT_DESC = (
    "Execute a bundled script from a loaded skill. Args are passed as a list (no "
    "shell). Returns stdout, stderr, and exit code. The skill must be loaded first."
)


def _error(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


def _ensure_loaded(context: Context, name: str):
    store = loaded_skills(context)
    if name not in store:
        raise LookupError(
            f"skill {name!r} is not loaded; call load_skill first"
        )
    return store[name]


# --- container mode: copy the skill into the task container, read/run there ---

def _container_mode(context: Context) -> bool:
    """True when skills are delivered INSIDE the task container (the driver sets
    context.metadata['skill_in_container']). Else host mode (read/run on the
    orchestrator host, the default — unchanged behaviour)."""
    return bool((context.metadata or {}).get("skill_in_container"))


async def _acquire_container(context: Context):
    """Acquire the rollout's container (same spec as the file tool, no mounts)."""
    image_id = (context.metadata or {}).get("image_id")
    if not image_id:
        raise LookupError("container-mode skills require context.metadata['image_id']")
    spec = ContainerResourceSpec(
        category="container", image=image_id,
        docker_host=(context.metadata or {}).get("docker_host"),
    )
    return await context.acquire_resource(id=context.rollout_id, spec=spec)


async def _stage_skill(context: Context, container, dest: str, host_dir) -> None:
    """Copy-on-demand: copy the skill into the container at `dest` once per rollout."""
    staged = context.metadata.setdefault("_staged_paths", set())
    if dest not in staged:
        await container.copy_in(str(host_dir), dest)
        staged.add(dest)


def _container_interpreter(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return "python3"
    if suffix in (".sh", ".bash"):
        return "bash"
    return ""  # executable script: run directly


async def _run_script_in_container(context, workdir, target, args, stdin, timeout):
    """Run a skill script inside the rollout container (so it uses the env's
    interpreter/deps + the task workspace). stderr is merged into stdout; the
    exit code is recovered from a trailing marker. Returns (output, exit_code,
    timed_out)."""
    container = await _acquire_container(context)
    interp = _container_interpreter(target)
    parts = ([interp, shlex.quote(target)] if interp else [shlex.quote(target)])
    invocation = " ".join(parts + [shlex.quote(a) for a in args])
    if stdin:
        b64 = base64.b64encode(stdin.encode("utf-8")).decode("ascii")
        invocation = f"printf %s {shlex.quote(b64)} | base64 -d | {invocation}"
    cmd = f"cd {shlex.quote(workdir)} && ( {invocation} ) 2>&1; echo \"__RC__$?\""
    try:
        out = await container.run_cmd(cmd, timeout=timeout)
    except asyncio.TimeoutError:
        return "", -1, True
    idx = out.rfind("__RC__")
    if idx < 0:
        return out, -1, False
    rc_tail = out[idx + len("__RC__"):].strip().splitlines()
    try:
        rc = int(rc_tail[0]) if rc_tail else -1
    except ValueError:
        rc = -1
    return out[:idx].rstrip("\n"), rc, False


@tool(name="load_skill", description=_LOAD_SKILL_DESC, max_length=16384)
async def load_skill(name: str, context: Context = None):
    """Load a skill into context.

    Args:
        name: The skill name from the available_skills listing. Must match exactly.
    """
    if context is None:
        return _error("context is required")
    try:
        root = find_skills_root(context)
        skill_dir = safe_skill_dir(root, name)
        parsed = parse_skill_md(skill_dir)
        manifest = build_manifest(skill_dir)
    except (FileNotFoundError, ValueError) as exc:
        return _error(str(exc))

    frontmatter = parsed["frontmatter"]
    canonical_name = frontmatter.get("name", name)
    version = frontmatter.get("version", "0.0.0")

    entry = {"dir": str(skill_dir), "files": manifest, "version": version}
    if _container_mode(context):
        # Copy the skill INTO the task container; read_skill_file/run_skill_script
        # then operate on that copy at $SKILL_DIR/<name> (manifest stays host-built).
        dest = f"{AF_CONTAINER_SKILLS_DIR}/{name}"
        try:
            container = await _acquire_container(context)
            await _stage_skill(context, container, dest, skill_dir)
        except Exception as exc:  # noqa: BLE001
            return _error(f"failed to stage skill into container: {exc}")
        entry = {"dir": dest, "host_dir": str(skill_dir), "files": manifest,
                 "version": version, "container": True}
    loaded_skills(context)[name] = entry

    return json.dumps(
        {
            "name": canonical_name,
            "version": version,
            "body": parsed["body"],
            "files": manifest,
        },
        ensure_ascii=False,
    )


@tool(name="read_skill_file", description=_READ_SKILL_FILE_DESC, max_length=_MAX_FILE_BYTES + 1024)
async def read_skill_file(skill: str, path: str, context: Context = None):
    """Read a file from a loaded skill.

    Args:
        skill: The skill name (must already be loaded).
        path: Relative path within the skill, e.g. references/plan-review.md. Must match an entry in the file manifest.
    """
    if context is None:
        return _error("context is required")
    try:
        entry = _ensure_loaded(context, skill)
    except LookupError as exc:
        return _error(str(exc))

    # Validate against the host copy (manifest was built host-side) for both modes.
    base = entry.get("host_dir", entry["dir"])
    try:
        resolved = resolve_manifest_path(Path(base), path, entry["files"])
    except (FileNotFoundError, ValueError) as exc:
        return _error(str(exc))

    if entry.get("container"):
        rel = resolved["real_path"].relative_to(base)
        target = f"{entry['dir']}/{rel}"
        try:
            container = await _acquire_container(context)
            out = await container.run_cmd(
                f"head -c {_MAX_FILE_BYTES + 1} -- {shlex.quote(target)}", timeout=60)
        except Exception as exc:  # noqa: BLE001
            return _error(f"failed to read skill file from container: {exc}")
        truncated = len(out.encode("utf-8", "replace")) > _MAX_FILE_BYTES
        content = out[:_MAX_FILE_BYTES] if truncated else out
    else:
        data = resolved["real_path"].read_bytes()
        truncated = len(data) > _MAX_FILE_BYTES
        if truncated:
            data = data[:_MAX_FILE_BYTES]
        content = data.decode("utf-8", errors="replace")

    return json.dumps(
        {"skill": skill, "path": path, "content": content, "truncated": truncated},
        ensure_ascii=False,
    )


def _script_command(real_path: Path, args: list) -> list:
    suffix = real_path.suffix.lower()
    if suffix == ".py":
        return [sys.executable or "python3", str(real_path), *args]
    if suffix in (".sh", ".bash"):
        return ["bash", str(real_path), *args]
    return [str(real_path), *args]


def _run_subprocess(cmd: list, cwd: str, stdin: str, timeout: int):
    try:
        completed = subprocess.run(
            cmd,
            input=stdin.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            timeout=timeout,
        )
        return completed.returncode, completed.stdout or b"", completed.stderr or b"", False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = (exc.stderr or b"") + f"\nTIMEOUT after {timeout}s\n".encode("utf-8")
        return -1, stdout, stderr, True


@tool(name="run_skill_script", description=_RUN_SKILL_SCRIPT_DESC, max_length=_MAX_STREAM_BYTES * 2 + 1024)
async def run_skill_script(
    skill: str,
    path: str,
    args: list = None,
    stdin: str = "",
    timeout_seconds: int = 30,
    context: Context = None,
):
    """Execute a bundled script from a loaded skill.

    Args:
        skill: The skill name (must already be loaded).
        path: Path to the script within the skill, e.g. scripts/safety_check.py. Must match a script entry in the manifest.
        args: Arguments passed to the script as argv. Each item is one argument; do not concatenate or shell-quote.
        stdin: Optional input piped to the script's stdin. Use this to pass content without writing to a file first.
        timeout_seconds: Max seconds for the subprocess (default 30, max 300).
    """
    if context is None:
        return _error("context is required")
    try:
        entry = _ensure_loaded(context, skill)
    except LookupError as exc:
        return _error(str(exc))

    # Validate against the host copy (manifest is host-built) for both modes.
    base = entry.get("host_dir", entry["dir"])
    try:
        resolved = resolve_manifest_path(Path(base), path, entry["files"])
    except (FileNotFoundError, ValueError) as exc:
        return _error(str(exc))

    if resolved["entry"].get("kind") != "script":
        return _error(f"path is not a script: {path}")

    arg_list = [str(a) for a in (args or [])]
    timeout = max(_MIN_TIMEOUT, min(int(timeout_seconds), _MAX_TIMEOUT))

    if entry.get("container"):
        # Run inside the rollout container so the script uses the env's deps +
        # the task workspace. stderr is merged into stdout.
        rel = resolved["real_path"].relative_to(base)
        target = f"{entry['dir']}/{rel}"
        try:
            out, exit_code, timed_out = await _run_script_in_container(
                context, entry["dir"], target, arg_list, stdin, timeout)
        except Exception as exc:  # noqa: BLE001
            return _error(f"failed to run skill script in container: {exc}")
        stdout_b, stderr_b = out.encode("utf-8", "replace"), b""
    else:
        cmd = _script_command(resolved["real_path"], arg_list)
        loop = asyncio.get_running_loop()
        exit_code, stdout_b, stderr_b, timed_out = await loop.run_in_executor(
            None, _run_subprocess, cmd, entry["dir"], stdin, timeout
        )

    stdout_truncated = len(stdout_b) > _MAX_STREAM_BYTES
    stderr_truncated = len(stderr_b) > _MAX_STREAM_BYTES
    if stdout_truncated:
        stdout_b = stdout_b[:_MAX_STREAM_BYTES]
    if stderr_truncated:
        stderr_b = stderr_b[:_MAX_STREAM_BYTES]

    return json.dumps(
        {
            "skill": skill,
            "path": path,
            "exit_code": exit_code,
            "stdout": stdout_b.decode("utf-8", errors="replace"),
            "stderr": stderr_b.decode("utf-8", errors="replace"),
            "timed_out": timed_out,
            "truncated": stdout_truncated or stderr_truncated,
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    print(load_skill.schema)
    print(read_skill_file.schema)
    print(run_skill_script.schema)
