"""Tests for MinisweAgent — mini-swe-agent behavior on AgentFly's rollout loop.

Two layers:

1. Pure-Python tests (run unconditionally): prompt construction against the
   exact strings the SkillsScrape miniswe SFT corpus carries, the observation
   JSON contract (mini-swe-agent 2.4.6's Jinja ``observation_template``
   including its HTML-safe ``tojson`` escaping), submit-marker detection,
   registry resolution, and the bash tool's no-container error path.

2. A live end-to-end rollout (``gpu`` marker): GPT-5.5 through the client
   backend drives a real container via the ``bash`` tool on a tiny
   create-a-file task and must finish by submitting. Skipped without an
   ``OPENAI_API_KEY`` or a usable local docker daemon.
"""

import asyncio
import functools
import json
import os
import shutil
import subprocess

import pytest

from agentfly.agents.specialized.miniswe import (
    MinisweAgent,
    build_instance_message,
    build_messages,
    build_system_prompt,
)
from agentfly.agents.specialized.miniswe.agent import SKILL_HEADER_SINGLE
from agentfly.agents.specialized.miniswe.prompts import (
    INSTANCE_BOILERPLATE,
    INSTANCE_PREFIX,
    SKILL_HEADER,
    SYSTEM_BASE,
    SYSTEM_INFORMATION_SENTINEL,
)
from agentfly.core.context import Context
from agentfly.tools import get_tools_from_names, miniswe_bash
from agentfly.tools.src.miniswe.tools import (
    OUTPUT_CAP,
    format_observation,
    is_submission,
)

MODEL = "gpt-5.5"
BASE_URL = "https://api.openai-next.com/v1"

SKILL = {
    "name": "wp-security",
    "description": "WordPress security best practices.",
    "directory": "wp-security",
}
SKILL2 = {
    "name": "csv-wrangler",
    "description": "Tabular data cleanup recipes.",
    "directory": "csv-wrangler",
}


@pytest.fixture(autouse=True)
def _dummy_openai_key(monkeypatch):
    """Constructing a ClientBackend needs a non-empty api_key; supply a
    placeholder for the pure-Python tests when none is set."""
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _build_agent():
    return MinisweAgent(
        model_name_or_path=MODEL,
        backend_config={
            "backend": "client",
            "base_url": BASE_URL,
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        monitors=[],
    )


# ---------------------------------------------------------------- prompts


def test_default_construction():
    agent = _build_agent()
    assert agent.tool_names == ["bash"]
    assert agent.system_prompt == SYSTEM_BASE
    # gpt-5.5 has no local HF tokenizer -> no local parser; tool calls come
    # from the OpenAI-compatible backend instead (parser-source fallback).
    assert agent.tool_parser is None


def test_qwen_model_resolves_tool_parser():
    """With a real Qwen tokenizer the qwen3_coder parser must resolve — this
    is the parsing path RL rollouts (verl/vLLM backends) rely on."""
    agent = MinisweAgent(
        model_name_or_path="Qwen/Qwen3.5-9B",
        backend_config={
            "backend": "client",
            "base_url": BASE_URL,
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        monitors=[],
    )
    assert agent.tool_parser is not None


def test_system_prompt_no_skill():
    assert build_system_prompt() == SYSTEM_BASE
    assert build_system_prompt([]) == SYSTEM_BASE


def test_system_prompt_single_skill_sft_format():
    """One skill renders the single-skill SFT-era layout (dash Location line) —
    the exact format 499/500 canonical miniswe SFT records carry."""
    expected = (
        "You are a helpful assistant that can interact with a computer.\n"
        "\n"
        "A Skill is available for this task:\n"
        "- wp-security: WordPress security best practices.\n"
        "- Location: /skills/wp-security/ (read its SKILL.md and follow it)"
    )
    assert build_system_prompt([SKILL]) == expected


def test_system_prompt_multi_skill_current_format():
    """Several skills render the current collection/eval layout (plural header,
    indented Location), preserving the given order (distractor ordering)."""
    prompt = build_system_prompt([SKILL, SKILL2])
    assert prompt.startswith(SYSTEM_BASE + "\n\n" + SKILL_HEADER + "\n")
    assert (
        "- wp-security: WordPress security best practices.\n"
        "  Location: /skills/wp-security/ (read its SKILL.md and follow it)\n"
        "- csv-wrangler: Tabular data cleanup recipes.\n"
        "  Location: /skills/csv-wrangler/ (read its SKILL.md and follow it)"
    ) in prompt
    assert prompt.index("wp-security") < prompt.index("csv-wrangler")


def test_system_prompt_style_override():
    single_forced = build_system_prompt([SKILL, SKILL2], style="single")
    assert SKILL_HEADER_SINGLE in single_forced
    multi_forced = build_system_prompt([SKILL], style="multi")
    assert SKILL_HEADER in multi_forced
    assert "  Location: /skills/wp-security/" in multi_forced


def test_instance_message():
    task = "Fix the bug in foo.py.\n"  # instruction.md keeps its newline
    uname = "Linux 6.12.0 #1 SMP x86_64"
    msg = build_instance_message(task, uname)
    assert msg.startswith(INSTANCE_PREFIX + task)
    # task's own trailing newline + the template's two = three total
    assert INSTANCE_PREFIX + task + "\n\n" + "You can execute bash commands" in msg
    assert f"<system_information>\n{uname}\n</system_information>" in msg
    assert SYSTEM_INFORMATION_SENTINEL not in msg
    assert msg.endswith(INSTANCE_BOILERPLATE.replace(SYSTEM_INFORMATION_SENTINEL, uname))


def test_build_messages_shape():
    msgs = build_messages("Do the thing.\n", skills=[SKILL])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "A Skill is available" in msgs[0]["content"]
    assert msgs[1]["content"].startswith(INSTANCE_PREFIX)


def test_preprocess_keeps_per_row_system_prompt():
    """Rows carry per-task/arm system messages; preprocessing must keep them
    (base behavior raises 'System prompt already exists') and only fall back
    to SYSTEM_BASE for rows without one."""
    agent = _build_agent()
    rows = [
        {"messages": build_messages("Do it.\n", skills=[SKILL]), "image_id": "img"},
        {"messages": [{"role": "user", "content": "hi"}]},
    ]
    out = agent._preprocess_messages(rows)
    sys0 = out[0]["messages"][0]
    assert sys0["role"] == "system"
    rendered = json.dumps(sys0["content"])
    assert "A Skill is available" in rendered  # row's own system kept, not replaced
    assert out[0]["image_id"] == "img"  # meta preserved
    sys1 = out[1]["messages"][0]
    assert sys1["role"] == "system"
    assert SYSTEM_BASE in json.dumps(sys1["content"])  # fallback applied


# ---------------------------------------------------------- observations


def test_format_observation_plain():
    assert format_observation(0, "hello\n") == (
        '{\n  "returncode": 0,\n  "output": "hello\\n"\n}'
    )


def test_format_observation_html_safe_escapes():
    """Jinja's ``tojson`` escapes < > & ' — the SFT corpus carries these."""
    obs = format_observation(0, "<a> & 'b'")
    assert '"output": "\\u003ca\\u003e \\u0026 \\u0027b\\u0027"' in obs
    # and it still parses back to the original string
    assert json.loads(obs)["output"] == "<a> & 'b'"


def test_format_observation_truncation_boundary():
    below = format_observation(0, "x" * (OUTPUT_CAP - 1))
    assert '"output":' in below and "output_head" not in below
    at_cap = format_observation(0, "x" * OUTPUT_CAP)  # template: length < 10000
    obj = json.loads(at_cap)
    assert obj["elided_chars"] == 0
    assert len(obj["output_head"]) == 5000 and len(obj["output_tail"]) == 5000
    assert obj["warning"] == "Output too long."


def test_format_observation_truncation_arithmetic():
    original = "h" * 6000 + "m" * 1000 + "t" * 6000
    obj = json.loads(format_observation(1, original))
    assert obj["output_head"] == original[:5000]
    assert obj["output_tail"] == original[-5000:]
    assert obj["elided_chars"] == len(original) - 10000


def test_format_observation_exception_info_same_line():
    obs = format_observation(-1, "", "Command 'sleep 99' timed out after 30 seconds")
    assert '"output": "", "exception_info": ' in obs  # same line, template trim
    assert json.loads(obs)["returncode"] == -1


def test_is_submission():
    assert is_submission("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n")
    assert is_submission("  MINI_SWE_AGENT_FINAL_OUTPUT")
    assert not is_submission("done: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
    assert not is_submission("")


# ------------------------------------------------------------------ tool


def test_tool_registry_resolution():
    (tool,) = get_tools_from_names(["bash"])
    assert tool.name == "bash"
    assert list(tool.args) == ["command"]
    assert tool is miniswe_bash or tool.name == miniswe_bash.name


def test_bash_tool_requires_image_id():
    ctx = Context(rollout_id="test-rollout", metadata={})
    result = asyncio.run(miniswe_bash(command="ls", context=ctx))
    assert result["status"] == "error"
    obs = json.loads(result["observation"])
    assert obs["returncode"] == 1
    assert "image_id" in obs["output"]


# ------------------------------------------------------------- live e2e


@functools.cache
def _docker_usable() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=15
            ).returncode
            == 0
        )
    except (subprocess.TimeoutExpired, OSError):
        return False


LIVE_IMAGE = os.getenv("MINISWE_TEST_IMAGE", "python:3.11-slim")


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_live_rollout_creates_file_and_submits():
    """GPT-5.5 solves a one-file task in a real container through the bash
    tool and must finish by echoing the submit marker."""
    if os.getenv("OPENAI_API_KEY", "test-key") == "test-key":
        pytest.skip("no OPENAI_API_KEY")

    agent = _build_agent()
    task = (
        "Create a file named answer.txt in your working directory containing "
        "exactly the text 42 and nothing else.\n"
    )
    messages = [
        {
            "messages": build_messages(task),
            "image_id": LIVE_IMAGE,
            # Engine resolution is meta -> AF_CONTAINER_ENGINE -> enroot
            # (cluster default). Pin docker explicitly so it matches this
            # test's docker-daemon skip guard.
            "container_engine": "enroot",
            "workdir": "/tmp",
            "exec_timeout": 60,
        }
    ]
    result = await agent.run(messages=messages, max_turns=8, num_chains=1)

    traj = result.trajectories[0]
    segment = traj.segments[0]
    commands = [
        json.loads(tc["function"]["arguments"])["command"]
        if isinstance(tc["function"]["arguments"], str)
        else tc["function"]["arguments"]["command"]
        for msg in segment
        if isinstance(msg, dict)
        for tc in msg.get("tool_calls") or []
    ]
    print(segment)
    assert commands, "model made no bash calls"
    assert any("answer.txt" in c for c in commands)
    assert any(is_submission(c.replace("echo ", "", 1)) or "COMPLETE_TASK" in c for c in commands)
    # tool observations must carry the miniswe JSON envelope. In trajectories
    # tool content is a list of parts: [{"type": "text", "text": <obs JSON>}].
    def _text(content):
        if isinstance(content, str):
            return content
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )

    tool_msgs = [m for m in segment if isinstance(m, dict) and m.get("role") == "tool"]
    assert tool_msgs
    assert all("returncode" in json.loads(_text(m["content"])) for m in tool_msgs)
