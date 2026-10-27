"""Skill object surfaced to the agent.

A ``Skill`` is what the agent's ``skills=`` argument accepts — it carries the
metadata that goes into the ``<available_skills>`` block in the system prompt
(spec §2.1). Skill *content* (SKILL.md body, scripts, references) is loaded
on demand via the ``load_skill`` tool; the agent only needs the name and
description to advertise availability.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .skill_loader import find_skills_root, parse_skill_md, safe_skill_dir


@dataclass
class Skill:
    name: str
    description: str
    version: str = "0.0.0"
    path: Optional[Path] = None

    @classmethod
    def from_dir(cls, skill_dir: Union[str, Path]) -> "Skill":
        skill_dir = Path(skill_dir)
        parsed = parse_skill_md(skill_dir)
        fm = parsed["frontmatter"]
        return cls(
            name=fm.get("name", skill_dir.name),
            description=fm.get("description", ""),
            version=fm.get("version", "0.0.0"),
            path=skill_dir,
        )


def load_skills(
    names: List[str],
    skills_root: Optional[Union[str, Path]] = None,
) -> List[Skill]:
    """Resolve a list of skill names to ``Skill`` objects.

    ``skills_root`` follows the same precedence as ``find_skills_root``:
    explicit arg → ``AF_SKILLS_ROOT`` env → repo default.
    """
    if skills_root is not None:
        root = Path(skills_root).expanduser().resolve()
    else:
        root = find_skills_root(None)
    return [Skill.from_dir(safe_skill_dir(root, name)) for name in names]
