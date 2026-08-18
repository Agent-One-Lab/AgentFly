"""End-to-end skills test against a Qwen3.5-4B server deployed with the
``qwen-skills`` chat template.

Skills reach the model via one of two mutually-exclusive paths:

1. Agent-layer (backend-agnostic) — when the system prompt carries a ``{skills}``
   slot, ``BaseAgent._render_skills_block()`` expands it into the system *message*
   content. Works for hosted models (OpenAI/Gemini) since it needs no template
   support. In this case ``_skills_payload()`` returns None to avoid double
   injection.
2. Template — when there is no ``{skills}`` slot, ``ChainRollout._skills_payload()``
   projects skills to ``[{"name", "description"}]`` dicts; ``ClientBackend`` re-emits
   them as ``extra_body={"chat_template_kwargs": {"skills": ...}}`` and the
   deployed chat template fills the skills block in the system prompt.

Either way the skill *body* (SKILL.md, scripts, references) is loaded on demand
via the ``load_skill`` / ``read_skill_file`` / ``run_skill_script`` tools. Each
GPU test prompt forces one of those code paths; assertions target the
trajectory's tool-call sequence and the final-answer text.
"""

import json
import os

import pytest

from agentfly.agents.specialized.hf_agent import HFAgent
# Side-effect import: registers the ``qwen3_coder`` vLLM tool parser, which
# understands Qwen 3.x's XML-style ``<tool_call><function=NAME>...</function></tool_call>``
# format (no system-prompt format instruction needed — it's the model default).
from agentfly.agents.specialized import swe_agents  # noqa: F401
from agentfly.tools import (
    load_skill,
    load_skills,
    read_skill_file,
    run_skill_script,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEST_SKILLS_ROOT = os.path.join(REPO_ROOT, "skills", "test-skills")


@pytest.fixture(autouse=True)
def _dummy_openai_key(monkeypatch):
    """The pure-Python plumbing tests build a ClientBackend, which needs a
    non-empty api_key at construction. Supply a placeholder when none is set so
    they run on CPU; the gpu-marked end-to-end tests use a real key/endpoint."""
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

MODEL = "gpt-5.5"
BASE_URL = "https://api.openai-next.com/v1"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a small set of bundled skills. "
    "Use load_skill to fetch a skill's SKILL.md and file manifest, then follow "
    "its instructions: use run_skill_script for scripts, read_skill_file for "
    "references."
)


def _tool_calls_in_segment(segment):
    """Return the list of tool names invoked across an entire trajectory segment."""
    names = []
    for msg in segment:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            if name:
                names.append(name)
    return names


def _final_assistant_text(segment) -> str:
    """Return the text of the last assistant message in the trajectory."""
    for msg in reversed(segment):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                )
    return ""


@pytest.fixture(scope="module", autouse=True)
def _set_skills_root():
    """Make the agent's skill resolver find ``skills/test-skills``."""
    prev = os.environ.get("AF_SKILLS_ROOT")
    os.environ["AF_SKILLS_ROOT"] = TEST_SKILLS_ROOT
    yield
    if prev is None:
        os.environ.pop("AF_SKILLS_ROOT", None)
    else:
        os.environ["AF_SKILLS_ROOT"] = prev


@pytest.fixture(scope="module")
def skills():
    return load_skills(
        ["hello-skill", "add-numbers", "country-capital", "word-count"],
        skills_root=TEST_SKILLS_ROOT,
    )


SYSTEM_PROMPT_WITH_SLOT = SYSTEM_PROMPT + "\n\n{skills}"


def _build_agent(skills, system_prompt=SYSTEM_PROMPT):
    return HFAgent(
        model_name_or_path=MODEL,
        tools=[load_skill, read_skill_file, run_skill_script],
        skills=skills,
        system_prompt=system_prompt,
        tool_parser_name="qwen3_coder",
        backend_config={
            "backend": "client",
            "base_url": BASE_URL,
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        monitors=[],
    )


def test_skills_payload_is_forwarded_to_backend(skills):
    """Pure-Python check: agent serializes skills into the dict shape vLLM expects.

    Runs unconditionally so a regression in the plumbing fails fast.
    """
    agent = _build_agent(skills)
    payload = agent._skills_payload()
    assert payload is not None
    assert len(payload) == len(skills)
    by_name = {s["name"]: s["description"] for s in payload}
    for skill in skills:
        assert skill.name in by_name
        assert by_name[skill.name] == skill.description


def test_skills_rendered_into_system_prompt_when_slot_present(skills):
    """With a ``{skills}`` slot the agent renders skills into the system message
    itself and suppresses the template payload (no double injection)."""
    agent = _build_agent(skills, system_prompt=SYSTEM_PROMPT_WITH_SLOT)

    # Template path is suppressed when the agent renders skills itself.
    assert agent._skills_payload() is None

    block = agent._render_skills_block()
    assert "<available_skills>" in block
    for skill in skills:
        assert skill.name in block
        assert skill.description in block

    rendered = json.dumps(
        agent._preprocess_messages([{"messages": [{"role": "user", "content": "hi"}]}])
    )
    assert "<available_skills>" in rendered
    assert "{skills}" not in rendered  # slot consumed, not left dangling
    for skill in skills:
        assert skill.name in rendered


def test_no_dangling_header_when_no_skills():
    """A ``{skills}`` slot with zero skills expands to empty — no dangling header."""
    agent = _build_agent([], system_prompt=SYSTEM_PROMPT_WITH_SLOT)
    assert agent._render_skills_block() == ""
    rendered = json.dumps(
        agent._preprocess_messages([{"messages": [{"role": "user", "content": "hi"}]}])
    )
    assert "<available_skills>" not in rendered
    assert "{skills}" not in rendered


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_country_capital_skill_canberra_trap(skills):
    """Model must consult ``country-capital`` to answer Australia → Canberra (not Sydney)."""
    agent = _build_agent(skills)
    messages = [
        {"messages": [{"role": "user", "content": "What is the capital of Australia?"}]}
    ]
    result = await agent.run(
        messages=messages, max_turns=6, num_chains=1
    )

    segment = result.trajectories[0].segments[0]
    invoked = _tool_calls_in_segment(segment)
    assert "load_skill" in invoked, f"expected load_skill, got: {invoked}"
    assert "read_skill_file" in invoked, f"expected read_skill_file, got: {invoked}"
    final = _final_assistant_text(segment)
    assert "Canberra" in final, f"expected 'Canberra' in final answer, got: {final!r}"


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_add_numbers_skill_runs_script(skills):
    """Model must use ``run_skill_script`` on add-numbers — not compute the sum itself."""
    agent = _build_agent(skills)
    messages = [{"messages": [{"role": "user", "content": "What is 17 plus 25?"}]}]
    result = await agent.run(
        messages=messages, max_turns=6, num_chains=1
    )

    segment = result.trajectories[0].segments[0]
    invoked = _tool_calls_in_segment(segment)
    assert "load_skill" in invoked, f"expected load_skill, got: {invoked}"
    assert "run_skill_script" in invoked, f"expected run_skill_script, got: {invoked}"
    final = _final_assistant_text(segment)
    assert "42" in final, f"expected '42' in final answer, got: {final!r}"


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_word_count_skill_exit_code_branch(skills):
    """Empty input → script exits 1 → model consults references/rules.md and explains."""
    agent = _build_agent(skills)
    messages = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Count the words in this text: \"This is a very short or very long context with some random words. Random words include terusg asdasf gaskg, and fmskg.\"."
                    ),
                }
            ]
        }
    ]
    result = await agent.run(
        messages=messages, max_turns=8, num_chains=1
    )

    segment = result.trajectories[0].segments[0]
    invoked = _tool_calls_in_segment(segment)
    # word-count SKILL.md instructs the model to consult references/rules.md when
    # the script exits non-zero, so all three skill tools should fire.
    assert "load_skill" in invoked
    assert "run_skill_script" in invoked
    assert "read_skill_file" in invoked, (
        f"expected read_skill_file on error branch, got: {invoked}"
    )
