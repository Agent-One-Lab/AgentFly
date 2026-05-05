"""Tests for run_shell_command tool with real container."""

import pytest

from agentfly.core import Context
from agentfly.tools.src.shell.tools import run_shell_command

IMAGE_ID = "swebench/swesmith.x86_64.andialbrecht_1776_sqlparse.e57923b3"


@pytest.mark.asyncio
async def test_run_shell_command_echo():
    context = Context(
        rollout_id="test_shell_echo",
        group_id="group_shell_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        result = (await run_shell_command(cmd="echo hello", context=context))['observation']
        print(f"observation: {result}")
        assert "hello" in result
    finally:
        await context.release_resource(scope="rollout")


@pytest.mark.asyncio
async def test_run_shell_command_pwd():
    context = Context(
        rollout_id="test_shell_pwd",
        group_id="group_shell_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        result = (await run_shell_command(cmd="pwd", context=context))['observation']
        print(f"observation: {result}")
        assert len(result.strip()) > 0
    finally:
        await context.release_resource(scope="rollout")


@pytest.mark.asyncio
async def test_run_shell_command_ls():
    context = Context(
        rollout_id="test_shell_ls",
        group_id="group_shell_tools",
        metadata={"image_id": IMAGE_ID},
    )
    try:
        result = (await run_shell_command(cmd="ls -la", context=context))['observation']
        print(f"observation: {result}")
    finally:
        await context.release_resource(scope="rollout")


@pytest.mark.asyncio
async def test_run_shell_command_requires_image_id():
    context = Context(
        rollout_id="test_shell_no_image",
        group_id="group_shell_tools",
        metadata={},
    )
    result = (await run_shell_command(cmd="echo x", context=context))['observation']
    print(f"observation: {result}")
    assert "Error" in result
    assert "image_id" in result
