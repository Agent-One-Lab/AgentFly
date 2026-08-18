"""Helpers shared by the three skill tools.

Skills live in folders that follow the layout described in
``skills/skill-infra-spec.md``:

```
<skills_root>/
└── <skill-name>/
    ├── SKILL.md          # required, YAML frontmatter + markdown body
    ├── scripts/          # bundled executables
    ├── references/       # docs loaded on demand
    └── assets/           # output templates / raw bytes
```

``find_skills_root`` resolves the root that contains skill folders.
Per-rollout state (which skills have been ``load_skill``-ed) is stored on
``context.metadata["_loaded_skills"]`` so the runtime can enforce
"must load before read/run" without a process-wide singleton.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# A skill name is one or more lowercase/digit/hyphen segments, separated by
# forward slashes. This allows simple names ("hello-skill") AND nested paths
# under the skills root ("microsoft/azure-skills/azure-upgrade"). Each segment
# is bounded to 64 chars; at most 8 nesting levels are accepted. ".." and
# uppercase are rejected by the segment alphabet itself.
_SKILL_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}(/[a-z0-9-]{1,64}){0,8}$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)
_MANIFEST_SUBFOLDERS = (("scripts", "script"), ("references", "reference"), ("assets", "asset"))


def find_skills_root(context: Optional[Any] = None) -> Path:
    """Resolve the directory that holds skill folders.

    Lookup order:
      1. ``context.metadata['skills_root']``
      2. ``AF_SKILLS_ROOT`` environment variable
      3. ``<repo>/skills/test-skills`` (located by walking up to the git root)
    """
    if context is not None:
        configured = context.metadata.get("skills_root")
        if configured:
            return Path(configured).expanduser().resolve()

    env_root = os.environ.get("AF_SKILLS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            candidate = parent / "skills" / "test-skills"
            if candidate.is_dir():
                return candidate.resolve()
            break

    raise FileNotFoundError(
        "skills root not found; set AF_SKILLS_ROOT or "
        "context.metadata['skills_root']"
    )


def safe_skill_dir(root: Path, name: str) -> Path:
    """Resolve ``root/name`` and verify it stays inside ``root``.

    ``name`` may be a simple slug (``"hello-skill"``) or a slash-separated
    path under the root (``"microsoft/azure-skills/azure-upgrade"``). Path
    traversal (``..``), uppercase, and other invalid characters are rejected
    by ``_SKILL_NAME_RE`` before any filesystem access.
    """
    if not _SKILL_NAME_RE.match(name):
        raise ValueError(f"invalid skill name: {name!r}")
    root_resolved = root.resolve()
    skill_dir = (root_resolved / name).resolve()
    try:
        skill_dir.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"skill {name!r} escapes skills root") from exc
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"skill not found: {name}")
    if not (skill_dir / "SKILL.md").is_file():
        raise FileNotFoundError(f"skill {name!r} is missing SKILL.md")
    return skill_dir


def parse_skill_md(skill_dir: Path) -> Dict[str, Any]:
    """Return ``{'frontmatter': dict, 'body': str}`` for ``SKILL.md``."""
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"SKILL.md is missing YAML frontmatter: {skill_dir}")
    frontmatter = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).lstrip("\n")
    return {"frontmatter": frontmatter, "body": body}


def _script_description(path: Path) -> str:
    """Best-effort one-line description for a bundled script."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    docstring = re.search(r'^(?:#![^\n]*\n)?\s*"""(.+?)"""', text, re.DOTALL)
    if docstring:
        first = docstring.group(1).strip().splitlines()
        if first:
            return first[0].strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#!"):
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            break
    return ""


def build_manifest(skill_dir: Path) -> List[Dict[str, Any]]:
    """Enumerate files the model is allowed to address via read_skill_file.

    Includes:
      * Top-level files at the skill root (e.g. SKILL.md, LICENSE,
        requirements.txt, package.json, pyproject.toml, setup.sh, ...).
        These are tagged ``kind="root"`` and are essential context for
        environment-building agents — without them, the agent has no way
        to read dependency declarations.
      * All files under the conventional scripts/references/assets
        subfolders, tagged with the corresponding ``kind``.

    Hidden files (dotfiles) and symlinks are excluded throughout. The
    SKILL.md body is also returned in full by ``load_skill``; the
    manifest entry is provided so an agent can re-read it via
    ``read_skill_file`` without erroring.
    """
    files: List[Dict[str, Any]] = []

    # Top-level files first, so the dependency declarations are visible
    # before the (typically much larger) script tree.
    try:
        root_entries = sorted(skill_dir.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        root_entries = []
    for path in root_entries:
        if not path.is_file() or path.is_symlink():
            continue
        if path.name.startswith("."):
            continue
        entry: Dict[str, Any] = {
            "path": path.name,
            "kind": "root",
            "size": path.stat().st_size,
        }
        files.append(entry)

    for sub, kind in _MANIFEST_SUBFOLDERS:
        sub_dir = skill_dir / sub
        if not sub_dir.is_dir():
            continue
        for path in sorted(sub_dir.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(skill_dir).as_posix()
            entry = {"path": rel, "kind": kind}
            if kind == "script":
                desc = _script_description(path)
                if desc:
                    entry["description"] = desc
            else:
                entry["size"] = path.stat().st_size
            files.append(entry)
    return files


def resolve_manifest_path(skill_dir: Path, rel_path: str, manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate ``rel_path`` against the manifest and return its entry + real path."""
    if not rel_path or rel_path.startswith("/") or "\\" in rel_path:
        raise ValueError(f"invalid path: {rel_path!r}")
    parts = rel_path.split("/")
    if any(part in ("", "..", ".") for part in parts):
        raise ValueError(f"invalid path: {rel_path!r}")

    skill_resolved = skill_dir.resolve()
    target = (skill_resolved / rel_path).resolve()
    try:
        target.relative_to(skill_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes skill: {rel_path!r}") from exc

    posix_rel = target.relative_to(skill_resolved).as_posix()
    entry = next((f for f in manifest if f["path"] == posix_rel), None)
    if entry is None:
        raise FileNotFoundError(f"path not in manifest: {rel_path}")
    if not target.is_file():
        raise FileNotFoundError(f"file does not exist: {rel_path}")
    # Reject symlinks that escape the skill directory.
    real = target.resolve(strict=True)
    try:
        real.relative_to(skill_resolved)
    except ValueError as exc:
        raise ValueError(f"symlink escapes skill: {rel_path!r}") from exc
    return {"entry": entry, "real_path": real}


def loaded_skills(context: Any) -> Dict[str, Dict[str, Any]]:
    """Return the per-rollout map of loaded skills, creating it on first access."""
    store = context.metadata.get("_loaded_skills")
    if store is None:
        store = {}
        context.metadata["_loaded_skills"] = store
    return store
