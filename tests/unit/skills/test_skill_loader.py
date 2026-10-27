"""Unit tests for the skill_loader helpers (no Context, no subprocess)."""

import os
from pathlib import Path

import pytest

from agentfly.tools.src.skills.skill_loader import (
    build_manifest,
    find_skills_root,
    parse_skill_md,
    resolve_manifest_path,
    safe_skill_dir,
)


def _write_skill(root: Path, name: str, body: str = "# body\n", version: str = "0.1.0"):
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\nversion: {version}\ndescription: test\n---\n\n{body}",
        encoding="utf-8",
    )
    return skill


def test_parse_skill_md_extracts_frontmatter_and_body(tmp_path):
    skill = _write_skill(tmp_path, "fm-test", body="# hello\nworld\n")
    parsed = parse_skill_md(skill)
    assert parsed["frontmatter"]["name"] == "fm-test"
    assert parsed["frontmatter"]["version"] == "0.1.0"
    assert parsed["body"].startswith("# hello")


def test_parse_skill_md_rejects_missing_frontmatter(tmp_path):
    skill = tmp_path / "no-fm"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# just markdown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frontmatter"):
        parse_skill_md(skill)


def test_build_manifest_orders_and_classifies(tmp_path):
    skill = _write_skill(tmp_path, "manifest-test")
    (skill / "scripts").mkdir()
    (skill / "scripts" / "go.py").write_text('"""Do the thing."""\nprint(1)\n', encoding="utf-8")
    (skill / "references").mkdir()
    (skill / "references" / "doc.md").write_text("# doc\n", encoding="utf-8")
    (skill / "assets").mkdir()
    (skill / "assets" / "tpl.txt").write_text("template", encoding="utf-8")

    manifest = build_manifest(skill)
    kinds = {f["path"]: f["kind"] for f in manifest}
    assert kinds == {
        "SKILL.md": "root",
        "scripts/go.py": "script",
        "references/doc.md": "reference",
        "assets/tpl.txt": "asset",
    }
    script_entry = next(f for f in manifest if f["kind"] == "script")
    assert script_entry["description"] == "Do the thing."
    assert "size" not in script_entry
    for kind in ("reference", "asset"):
        entry = next(f for f in manifest if f["kind"] == kind)
        assert entry["size"] > 0


def test_build_manifest_skips_symlinks(tmp_path):
    skill = _write_skill(tmp_path, "sym-test")
    (skill / "references").mkdir()
    real = skill / "references" / "real.md"
    real.write_text("real", encoding="utf-8")
    link = skill / "references" / "link.md"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    manifest = build_manifest(skill)
    paths = [f["path"] for f in manifest]
    assert "references/real.md" in paths
    assert "references/link.md" not in paths


def test_safe_skill_dir_validates_name(tmp_path):
    _write_skill(tmp_path, "ok-name")
    assert safe_skill_dir(tmp_path, "ok-name").name == "ok-name"
    with pytest.raises(ValueError, match="invalid skill name"):
        safe_skill_dir(tmp_path, "../oops")
    with pytest.raises(ValueError, match="invalid skill name"):
        safe_skill_dir(tmp_path, "Has-Caps")
    with pytest.raises(FileNotFoundError):
        safe_skill_dir(tmp_path, "no-such-skill")


def test_safe_skill_dir_requires_skill_md(tmp_path):
    incomplete = tmp_path / "missing-md"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError, match="SKILL.md"):
        safe_skill_dir(tmp_path, "missing-md")


def test_resolve_manifest_path_accepts_listed_file(tmp_path):
    skill = _write_skill(tmp_path, "rmp")
    (skill / "references").mkdir()
    target = skill / "references" / "a.md"
    target.write_text("a", encoding="utf-8")
    manifest = build_manifest(skill)
    resolved = resolve_manifest_path(skill, "references/a.md", manifest)
    assert resolved["real_path"] == target.resolve()
    assert resolved["entry"]["kind"] == "reference"


def test_resolve_manifest_path_rejects_unlisted(tmp_path):
    skill = _write_skill(tmp_path, "rmp2")
    # A file under a non-scanned subfolder (not scripts/references/assets) never
    # enters the manifest, so resolving it must fail.
    (skill / "notes").mkdir()
    (skill / "notes" / "secret.txt").write_text("nope", encoding="utf-8")
    manifest = build_manifest(skill)
    with pytest.raises(FileNotFoundError, match="manifest"):
        resolve_manifest_path(skill, "notes/secret.txt", manifest)


@pytest.mark.parametrize("bad", ["..", "../", "/abs", "ref/../etc", "", "ref\\win"])
def test_resolve_manifest_path_rejects_bad_inputs(tmp_path, bad):
    skill = _write_skill(tmp_path, "rmp3")
    (skill / "references").mkdir()
    (skill / "references" / "a.md").write_text("a", encoding="utf-8")
    manifest = build_manifest(skill)
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_manifest_path(skill, bad, manifest)


def test_find_skills_root_from_context_metadata(tmp_path):
    class StubCtx:
        metadata = {"skills_root": str(tmp_path)}

    assert find_skills_root(StubCtx()) == tmp_path.resolve()


def test_find_skills_root_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AF_SKILLS_ROOT", str(tmp_path))
    assert find_skills_root(None) == tmp_path.resolve()


def test_find_skills_root_context_beats_env(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("AF_SKILLS_ROOT", str(other))

    class StubCtx:
        metadata = {"skills_root": str(tmp_path)}

    assert find_skills_root(StubCtx()) == tmp_path.resolve()
