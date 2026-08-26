"""MinisweAgent — mini-swe-agent's behavior as an AgentFly agent.

Reproduces the exact prompt + tool contract of the SkillsScrape miniswe
trajectory collection (mini-swe-agent in native tool-calling mode), so that a
model SFT'd on those trajectories rolls out on-policy during RL training:

* system prompt: ``SYSTEM_BASE`` alone (no-skill arm), or with the skill
  header + per-skill block (skill arms; skills are real files under
  ``/skills/<dir>/`` in the task container),
* user turn: ``Please solve this issue: {task}`` + the verbatim mini-swe
  instance boilerplate (workflow, command-execution rules, examples),
* one native tool: ``bash`` (fresh subshell per call, JSON observations),
  multiple calls per assistant turn allowed,
* termination: ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` (the bash tool
  returns ``status="terminal"``), a no-tool-call turn, or ``max_turns``.

Prompt constants live in :mod:`.prompts`, a GENERATED module extracted
byte-for-byte from real SFT records (SkillsScrape
``scripts/rl/gen_miniswe_prompts.py``). Message construction for dataset rows
goes through :func:`build_system_prompt` / :func:`build_instance_message` so
the RL dataset builder and the agent cannot drift apart.

Per-task/arm prompts vary, so dataset rows should carry their own rendered
``system`` message (built with the helpers here); the agent-level
``system_prompt`` is only the fallback for rows without one. Container
binding rides on ``Messages.meta``: ``image_id`` (required), ``docker_host``,
``workdir``, ``exec_timeout``, ``environment`` — see the bash tool's
docstring.
"""

from typing import Dict, List, Optional

from ....tools.src.miniswe import miniswe_bash
from ...agent_base import BaseAgent
from .prompts import (
    INSTANCE_BOILERPLATE,
    INSTANCE_PREFIX,
    INSTANCE_SEPARATOR,
    SKILL_HEADER,
    SYSTEM_BASE,
    SYSTEM_INFORMATION_SENTINEL,
)

SKILLS_ROOT = "/skills"


def _oneline(value: str) -> str:
    """Collapse whitespace and neutralize Jinja-ish braces (mirrors the
    collection stack's ``harbor_agent._fill`` value sanitization)."""
    return " ".join(str(value).split()).replace("{{", "{ {").replace("}}", "} }")


# Header + block layout of the single-skill collection era — the format
# 499/500 canonical miniswe SFT records carry. Note the "- Location:" dash.
SKILL_HEADER_SINGLE = "A Skill is available for this task:"


def render_skill_block(
    skills: List[Dict[str, str]], root: str = SKILLS_ROOT, style: str = "multi"
) -> str:
    """Render the per-skill lines.

    ``style="multi"``: the current collection/eval format
    (``harbor_agent._fill``) — ``  Location:`` indented under each entry.
    ``style="single"``: the single-skill SFT-era format — ``- Location:``
    as a sibling dash line.

    Each entry: ``{"name": ..., "description": ..., "directory": ...}``
    (``directory`` defaults to ``name``). Order is preserved — the collection
    deliberately keeps distractor ordering as given.
    """
    root = _oneline(root).rstrip("/")
    location_prefix = "- " if style == "single" else "  "
    lines = []
    for skill in skills:
        name = _oneline(skill.get("name") or skill.get("directory") or "")
        description = _oneline(skill.get("description", ""))
        directory = _oneline(skill.get("directory") or skill.get("name") or "")
        lines.append(
            f"- {name}: {description}\n"
            f"{location_prefix}Location: {root}/{directory}/ (read its SKILL.md and follow it)"
        )
    return "\n".join(lines)


def build_system_prompt(
    skills: Optional[List[Dict[str, str]]] = None,
    root: str = SKILLS_ROOT,
    style: str = "auto",
) -> str:
    """System prompt for an arm: base line alone (no skills) or with a skill
    header + block.

    ``style="auto"`` (default) renders one skill in the single-skill SFT-era
    format (what the model was trained on) and several skills in the current
    multi-skill format (what distractor arms and the current eval use).
    Pass ``"single"`` or ``"multi"`` to force either.
    """
    if not skills:
        return SYSTEM_BASE
    if style == "auto":
        style = "single" if len(skills) == 1 else "multi"
    if style == "single":
        header = SKILL_HEADER_SINGLE
    else:
        header = SKILL_HEADER
    return f"{SYSTEM_BASE}\n\n{header}\n{render_skill_block(skills, root, style)}"


def build_instance_message(task: str, system_information: str = "") -> str:
    """User turn: prefix + task VERBATIM (keep instruction.md's trailing
    newline) + separator + boilerplate with the container uname filled in."""
    boilerplate = INSTANCE_BOILERPLATE.replace(
        SYSTEM_INFORMATION_SENTINEL, system_information
    )
    return f"{INSTANCE_PREFIX}{task}{INSTANCE_SEPARATOR}{boilerplate}"


def build_messages(
    task: str,
    skills: Optional[List[Dict[str, str]]] = None,
    system_information: str = "",
    root: str = SKILLS_ROOT,
    style: str = "auto",
) -> List[Dict[str, str]]:
    """Full initial message list for one task/arm."""
    return [
        {"role": "system", "content": build_system_prompt(skills, root, style)},
        {"role": "user", "content": build_instance_message(task, system_information)},
    ]


class MinisweAgent(BaseAgent):
    """mini-swe-agent behavior (prompts + single ``bash`` tool) on AgentFly's
    rollout loop. See the module docstring for the contract."""

    def __init__(
        self,
        model_name_or_path: str,
        tools: List = None,
        system_prompt: str = None,
        tool_parser_name: Optional[str] = "qwen3_coder",
        **kwargs,
    ):
        # Ensure the qwen3_coder tool parser is registered before BaseAgent
        # resolves tool_parser_name (same side-effect import the SWE agents use).
        from .. import swe_agents  # noqa: F401

        super().__init__(
            model_name_or_path=model_name_or_path,
            tools=tools if tools is not None else [miniswe_bash],
            system_prompt=system_prompt or SYSTEM_BASE,
            tool_parser_name=tool_parser_name,
            **kwargs,
        )

    def _preprocess_messages(self, messages):
        """Per-task/arm rows carry their OWN system message (arms differ within
        one batch, so a single agent-level prompt cannot express them). The base
        implementation raises when a row already has a system message and the
        agent has ``system_prompt`` set; here the agent-level prompt is only a
        per-row FALLBACK for rows that come without one."""
        from ...utils.messages import MessagesList

        messages_list = MessagesList.from_data(messages)
        for msgs in messages_list:
            turns = msgs.messages
            if not (turns and turns[0].get("role") == "system"):
                msgs.set_system_prompt(self.system_prompt or SYSTEM_BASE)
        return messages_list.to_list()
