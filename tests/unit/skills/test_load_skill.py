import pytest

from agentfly.tools.src.skills import load_skill


def test_load_skill_schema():
    schema = load_skill.schema["function"]
    print(f"schema: {schema}")
    assert schema["name"] == "load_skill"
    assert "Load a skill" in schema["description"]
    assert schema["parameters"]["required"] == ["name"]
    assert "context" not in schema["parameters"]["properties"]


@pytest.mark.asyncio(loop_scope="session")
async def test_load_skill_no_subfolders(context, parse_observation):
    """hello-skill has only SKILL.md → empty manifest, body delivered."""
    result = parse_observation(await load_skill(name="hello-skill", context=context))
    print(f"result: {result}")
    assert result["name"] == "hello-skill"
    assert result["version"] == "0.1.0"
    assert "Hello from skill!" in result["body"]
    # Only SKILL.md at the root (a kind="root" entry); no subfolder files.
    assert [f for f in result["files"] if f["kind"] != "root"] == []


@pytest.mark.asyncio(loop_scope="session")
async def test_load_skill_with_script(context, parse_observation):
    """add-numbers contributes one script entry with a description."""
    result = parse_observation(await load_skill(name="add-numbers", context=context))
    print(f"result: {result}")
    scripts = [f for f in result["files"] if f["kind"] == "script"]
    assert len(scripts) == 1
    entry = scripts[0]
    assert entry["path"] == "scripts/add.py"
    assert entry["kind"] == "script"
    assert "two integers" in entry["description"].lower()
    assert "size" not in entry


@pytest.mark.asyncio(loop_scope="session")
async def test_load_skill_with_reference(context, parse_observation):
    """country-capital contributes a reference entry with a size."""
    result = parse_observation(await load_skill(name="country-capital", context=context))
    print(f"result: {result}")
    entry = next(f for f in result["files"] if f["kind"] == "reference")
    assert entry["path"] == "references/capitals.md"
    assert entry["size"] > 0
    assert "description" not in entry


@pytest.mark.asyncio(loop_scope="session")
async def test_load_skill_mixed_manifest(context, parse_observation):
    """word-count has both a script and a reference; both surface."""
    result = parse_observation(await load_skill(name="word-count", context=context))
    kinds = sorted(f["kind"] for f in result["files"] if f["kind"] != "root")
    print(f"kinds: {kinds}")
    assert kinds == ["reference", "script"]


@pytest.mark.asyncio(loop_scope="session")
async def test_load_skill_marks_state(context):
    """After load_skill, the rollout context records the load for downstream tools."""
    await load_skill(name="hello-skill", context=context)
    loaded = context.metadata["_loaded_skills"]
    print(f"loaded: {loaded}")
    assert "hello-skill" in loaded
    assert loaded["hello-skill"]["version"] == "0.1.0"
    assert "dir" in loaded["hello-skill"]


@pytest.mark.asyncio(loop_scope="session")
async def test_load_skill_unknown(context, parse_observation):
    result = parse_observation(await load_skill(name="does-not-exist", context=context))
    assert "error" in result
    assert "not found" in result["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_load_skill_invalid_name(context, parse_observation):
    """Names must match [a-z0-9-]{1,64}; traversal-like names are rejected up front."""
    result = parse_observation(await load_skill(name="../etc", context=context))
    print(f"result: {result}")
    assert "error" in result
    assert "invalid skill name" in result["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_load_skill_uppercase_name_rejected(context, parse_observation):
    result = parse_observation(await load_skill(name="Hello-Skill", context=context))
    print(f"result: {result}")
    assert "error" in result
    assert "invalid skill name" in result["error"]
