import pytest

from agentfly.tools.src.skills import load_skill, run_skill_script


def test_run_skill_script_schema():
    schema = run_skill_script.schema["function"]
    assert schema["name"] == "run_skill_script"
    assert "shell" in schema["description"].lower()
    required = schema["parameters"]["required"]
    assert "skill" in required and "path" in required
    # args/stdin/timeout are optional
    for opt in ("args", "stdin", "timeout_seconds"):
        assert opt in schema["parameters"]["properties"]
        assert opt not in required
    args_prop = schema["parameters"]["properties"]["args"]
    assert args_prop["type"] == "array"
    assert args_prop["items"] == {"type": "string"}


@pytest.mark.asyncio(loop_scope="session")
async def test_run_skill_script_stdin_round_trip(context, parse_observation):
    await load_skill(name="add-numbers", context=context)
    result = parse_observation(
        await run_skill_script(
            skill="add-numbers", path="scripts/add.py", stdin="3 5", context=context
        )
    )
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "8"
    assert result["stderr"] == ""
    assert result["timed_out"] is False
    assert result["truncated"] is False


@pytest.mark.asyncio(loop_scope="session")
async def test_run_skill_script_non_zero_exit_surfaces_stderr(context, parse_observation):
    await load_skill(name="word-count", context=context)
    result = parse_observation(
        await run_skill_script(
            skill="word-count", path="scripts/count.py", stdin="", context=context
        )
    )
    assert result["exit_code"] == 1
    assert "empty input" in result["stderr"]
    assert result["stdout"] == ""


@pytest.mark.asyncio(loop_scope="session")
async def test_run_skill_script_requires_load(context, parse_observation):
    result = parse_observation(
        await run_skill_script(
            skill="add-numbers", path="scripts/add.py", stdin="1 2", context=context
        )
    )
    assert "error" in result
    assert "load_skill" in result["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_run_skill_script_rejects_non_script_kind(context, parse_observation):
    """References are readable but not runnable."""
    await load_skill(name="country-capital", context=context)
    result = parse_observation(
        await run_skill_script(
            skill="country-capital",
            path="references/capitals.md",
            context=context,
        )
    )
    assert "error" in result
    assert "not a script" in result["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_run_skill_script_rejects_non_manifest(context, parse_observation):
    await load_skill(name="add-numbers", context=context)
    result = parse_observation(
        await run_skill_script(
            skill="add-numbers", path="scripts/missing.py", context=context
        )
    )
    assert "error" in result
    assert "manifest" in result["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_run_skill_script_clamps_timeout(context, parse_observation):
    """timeout_seconds is clamped to [1, 300]; oversized values must not raise."""
    await load_skill(name="add-numbers", context=context)
    result = parse_observation(
        await run_skill_script(
            skill="add-numbers",
            path="scripts/add.py",
            stdin="1 2",
            timeout_seconds=99999,
            context=context,
        )
    )
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "3"


@pytest.mark.asyncio(loop_scope="session")
async def test_run_skill_script_times_out(tmp_path, parse_observation):
    """A sleep-loop script trips the timeout path."""
    import shutil
    from agentfly.core import Context

    sandbox_root = tmp_path / "sk"
    skill_dir = sandbox_root / "slow-skill"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: slow-skill\nversion: 0.1.0\ndescription: blocks forever\n---\n\n# slow\n",
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "sleep.py").write_text(
        "import time\nwhile True:\n    time.sleep(1)\n",
        encoding="utf-8",
    )

    ctx = Context(rollout_id="slow", metadata={"skills_root": str(sandbox_root)})
    await load_skill(name="slow-skill", context=ctx)
    result = parse_observation(
        await run_skill_script(
            skill="slow-skill",
            path="scripts/sleep.py",
            timeout_seconds=1,
            context=ctx,
        )
    )
    assert result["timed_out"] is True
    assert result["exit_code"] == -1
    shutil.rmtree(sandbox_root, ignore_errors=True)
