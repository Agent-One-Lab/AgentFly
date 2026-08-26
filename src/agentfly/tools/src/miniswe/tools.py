"""mini-swe-agent style ``bash`` tool for :class:`MinisweAgent`.

Reproduces the observation contract of the SkillsScrape miniswe trajectory
collection (mini-swe-agent in native tool-calling mode) so RL rollouts match
the SFT distribution byte-for-byte:

* Each command runs in a FRESH ``bash -c`` subshell (no state persists between
  calls), cwd = the task workspace, stderr merged into stdout.
* The observation is a JSON string:
      {"returncode": <int>, "output": <str>}                     (indent=2)
  or, when the output exceeds ``OUTPUT_CAP`` characters:
      {"returncode": ..., "output_head": <first 5000>,
       "output_tail": <last 5000>, "elided_chars": <n>,
       "warning": "Output too long."}
* The episode ends when a command's output starts with the submit marker
  (the model runs ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``): the tool
  returns ``status="terminal"`` which the chain loop maps to a terminal node.

Deliberately NO CommandFilter: the collection ran unfiltered commands inside
isolated task containers, and a filter would perturb the observation
distribution relative to SFT.

Context metadata consumed (all set by the caller via ``Messages.meta``):
  image_id (required), docker_host, workdir (default /workspace),
  exec_timeout (seconds, default 300), environment.
"""

import asyncio
import json
import re
import shlex

from ....core import Context
from ....resources import ContainerResourceSpec
from ...decorator import tool
from ...types import ToolResult

WORKSPACE_DIR = "/workspace"
# Per-command timeout. 30s matches the trajectory-collection environment
# (its timeout observations read "timed out after 30 seconds"); override
# per task via metadata["exec_timeout"].
EXEC_TIMEOUT_DEFAULT = 30

# Observation truncation contract (recovered from the SFT corpus).
OUTPUT_CAP = 10000
HEAD_CHARS = 5000
TAIL_CHARS = 5000
TRUNCATION_WARNING = "Output too long."

SUBMIT_MARKERS = (
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    "MINI_SWE_AGENT_FINAL_OUTPUT",
)

_RC_MARKER = "__MINISWE_RC__"
_RC_RE = re.compile(rf"\n?{_RC_MARKER}(-?\d+)\s*$")


def _tojson(value: str) -> str:
    """Jinja's ``| tojson`` filter: json.dumps + HTML-safe escapes.

    mini-swe-agent renders observations through a Jinja template
    (config/mini.yaml ``observation_template``), so strings carry Jinja's
    HTML-safe escaping of ``<``, ``>``, ``&``, ``'`` — reproduced here so
    observations match the SFT corpus byte-for-byte.
    """
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )


def format_observation(returncode: int, output: str, exception_info: str = None) -> str:
    """Render the observation exactly as mini-swe-agent 2.4.6's
    ``observation_template`` does (see :func:`_tojson`). Truncation kicks in
    at ``len(output) >= 10000`` (the template's ``| length < 10000`` branch);
    ``exception_info``, when present, rides on the same line as the last
    output field (the template's ``{%- if %}`` trimming)."""
    exc = f', "exception_info": {_tojson(exception_info)}' if exception_info else ""
    if len(output) < OUTPUT_CAP:
        return (
            "{\n"
            f'  "returncode": {returncode},\n'
            f'  "output": {_tojson(output)}{exc}\n'
            "}"
        )
    return (
        "{\n"
        f'  "returncode": {returncode},\n'
        f'  "output_head": {_tojson(output[:HEAD_CHARS])},\n'
        f'  "output_tail": {_tojson(output[-TAIL_CHARS:])},\n'
        f'  "elided_chars": {len(output) - OUTPUT_CAP},\n'
        f'  "warning": "{TRUNCATION_WARNING}"{exc}\n'
        "}"
    )


def is_submission(output: str) -> bool:
    stripped = output.lstrip()
    return any(stripped.startswith(m) for m in SUBMIT_MARKERS)


@tool(name="bash", max_length=1_000_000)
async def miniswe_bash(command: str, context: Context):
    """
    Execute a bash command in the task container.

    Args:
        command (str): The bash command to execute. Runs in a fresh subshell
            in the task workspace; directory and environment-variable changes
            do not persist between calls.
        context (Context): Injected rollout context; used to acquire the
            per-rollout task container.
    """
    image_id = context.metadata.get("image_id")
    if not image_id:
        return ToolResult(
            name="bash",
            arguments={"command": command},
            observation=format_observation(
                1, "Error: context.metadata['image_id'] is required for the bash tool."
            ),
            status="error",
        )

    spec = ContainerResourceSpec(
        category="container",
        image=image_id,
        docker_host=context.metadata.get("docker_host"),
    )
    container = await context.acquire_resource(
        id=context.rollout_id,
        spec=spec,
        backend=context.resource_backend,
        timeout=1200,
    )

    workdir = context.metadata.get("workdir") or WORKSPACE_DIR
    timeout = context.metadata.get("exec_timeout") or EXEC_TIMEOUT_DEFAULT

    # Fresh `bash -c` per call (mini-swe semantics), stderr merged, and the
    # subshell's exit code smuggled out through a trailing marker because
    # run_cmd only returns the output stream.
    wrapped = (
        f"bash -c {shlex.quote(command)} 2>&1; "
        f"printf '\\n{_RC_MARKER}%d' $?"
    )
    try:
        raw = await container.run_cmd(wrapped, timeout=timeout, workdir=workdir)
    except asyncio.TimeoutError:
        # Mirror the collection environment's timeout observation.
        return ToolResult(
            name="bash",
            arguments={"command": command},
            observation=format_observation(
                -1,
                "",
                exception_info=(
                    "An error occurred while executing the command: "
                    f"Command '{command}' timed out after {timeout} seconds"
                ),
            ),
            status="success",
        )

    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
    m = _RC_RE.search(text)
    if m:
        returncode = int(m.group(1))
        output = text[: m.start()]
    else:
        # Marker lost (e.g. the command killed its own shell); report what we saw.
        returncode = -1
        output = text

    return ToolResult(
        name="bash",
        arguments={"command": command},
        observation=format_observation(returncode, output),
        status="terminal" if is_submission(output) else "success",
    )
