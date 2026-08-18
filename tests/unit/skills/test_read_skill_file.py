import pytest

from agentfly.tools.src.skills import load_skill, read_skill_file


def test_read_skill_file_schema():
    schema = read_skill_file.schema["function"]
    print(f"schema: {schema}")
    assert schema["name"] == "read_skill_file"
    assert "loaded" in schema["description"].lower()
    assert schema["parameters"]["required"] == ["skill", "path"]


@pytest.mark.asyncio(loop_scope="session")
async def test_read_skill_file_returns_content(context, parse_observation):
    await load_skill(name="country-capital", context=context)
    result = parse_observation(
        await read_skill_file(skill="country-capital", path="references/capitals.md", context=context)
    )
    print(f"result: {result}")
    assert result["skill"] == "country-capital"
    assert result["path"] == "references/capitals.md"
    assert "Canberra" in result["content"]
    assert result["truncated"] is False


@pytest.mark.asyncio(loop_scope="session")
async def test_read_skill_file_requires_load(context, parse_observation):
    """Reading before load_skill must fail with a load_skill-pointing error."""
    result = parse_observation(
        await read_skill_file(skill="country-capital", path="references/capitals.md", context=context)
    )
    print(f"result: {result}")
    assert "error" in result
    assert "load_skill" in result["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_read_skill_file_rejects_traversal(context, parse_observation):
    await load_skill(name="country-capital", context=context)
    result = parse_observation(
        await read_skill_file(skill="country-capital", path="../hello-skill/SKILL.md", context=context)
    )
    assert "error" in result
    assert "invalid path" in result["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_read_skill_file_rejects_absolute(context, parse_observation):
    await load_skill(name="country-capital", context=context)
    result = parse_observation(
        await read_skill_file(skill="country-capital", path="/etc/passwd", context=context)
    )
    print(f"result: {result}")
    assert "error" in result
    assert "invalid path" in result["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_read_skill_file_rejects_non_manifest(context, parse_observation):
    """A path that is not a manifest entry is rejected — only listed files are readable."""
    await load_skill(name="country-capital", context=context)
    result = parse_observation(
        await read_skill_file(skill="country-capital", path="references/not-listed.md", context=context)
    )
    print(f"result: {result}")
    assert "error" in result
    assert "manifest" in result["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_read_skill_file_accepts_script_path(context, parse_observation):
    """Manifest entries of any kind are readable — kind only restricts run_skill_script."""
    await load_skill(name="add-numbers", context=context)
    result = parse_observation(
        await read_skill_file(skill="add-numbers", path="scripts/add.py", context=context)
    )
    print(f"result: {result}")
    assert "stdin.read" in result["content"]


@pytest.mark.asyncio(loop_scope="session")
async def test_read_skill_file_load_state_is_per_rollout(context, test_skills_root, parse_observation):
    """Loading in one Context must not satisfy read_skill_file in another."""
    from agentfly.core import Context

    await load_skill(name="country-capital", context=context)
    other = Context(rollout_id="other", metadata={"skills_root": str(test_skills_root)})
    result = parse_observation(
        await read_skill_file(skill="country-capital", path="references/capitals.md", context=other)
    )
    print(f"result: {result}")
    assert "error" in result
    assert "load_skill" in result["error"]
