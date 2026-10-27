"""Skills: reusable instruction bundles an agent can load on demand.

A :class:`Skill` is a folder with a ``SKILL.md`` (name, description, instructions) plus optional
scripts and reference files. Agents advertise skills by name and description; the tools
``load_skill``, ``read_skill_file`` and ``run_skill_script`` load the rest when the model asks.
"""

from .skill import Skill, load_skills
from .tools import load_skill, read_skill_file, run_skill_script

__all__ = [
    "Skill",
    "load_skill",
    "load_skills",
    "read_skill_file",
    "run_skill_script",
]
