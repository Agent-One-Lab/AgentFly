"""Unit tests for the unified loop-control policy shared by chain and step rollouts.

Covers the three run knobs (on_no_tool_call / on_invalid_tool_call / on_tool_error),
the tool-emitted ``control`` override, the status diagnoses that route to them, and the
tool-layer producers (submit_tool_call → status="invalid"; a tool-body exception →
status="error"; a dict ``control`` / ``status`` round-tripping through ToolResult).
"""

import pytest

from agentfly.tools.types import ToolResult, CONTROL
from agentfly.tools.tool_base import submit_tool_call, InvalidToolCallError
from agentfly.tools.decorator import tool
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.rollout.strategies.step_rollout import StepRollout
from agentfly.agents.rollout.events import FinishReason


def _res(status="success", control=None):
    return ToolResult(name="t", arguments={}, observation="x", status=status, control=control)


# ---- defaults + validation -------------------------------------------------

@pytest.mark.parametrize("cls,kw", [(ChainRollout, {}), (StepRollout, {"prompt_builder": "alfworld_flat"})])
def test_default_policy(cls, kw):
    r = cls(**kw)
    assert r.on_no_tool_call == "end"
    assert r.on_invalid_tool_call == "end"
    assert r.on_tool_error == "continue"


def test_invalid_policy_value_rejected():
    with pytest.raises(ValueError):
        ChainRollout(on_tool_error="nope")
    with pytest.raises(ValueError):
        StepRollout(prompt_builder="alfworld_flat", on_no_tool_call="stop")


# ---- resolve_tool_control precedence --------------------------------------

def test_success_continues():
    assert ChainRollout().resolve_tool_control(_res("success")) == "continue"


def test_invalid_routes_to_policy():
    r = ChainRollout(on_invalid_tool_call="end")
    assert r.resolve_tool_control(_res("invalid")) == "end"
    assert ChainRollout(on_invalid_tool_call="raise").resolve_tool_control(_res("invalid")) == "raise"


def test_error_routes_to_policy():
    assert ChainRollout(on_tool_error="continue").resolve_tool_control(_res("error")) == "continue"
    assert ChainRollout(on_tool_error="end").resolve_tool_control(_res("error")) == "end"


def test_tool_control_wins_over_policy():
    # Even though the default on_tool_error=continue, an explicit control ends.
    assert ChainRollout().resolve_tool_control(_res("error", control="end")) == "end"
    # And a tool can force continue past an invalid/error diagnosis.
    assert ChainRollout(on_invalid_tool_call="raise").resolve_tool_control(
        _res("invalid", control="continue")
    ) == "continue"


def test_message_ends_episode():
    r = ChainRollout()
    assert r.message_ends_episode({"status": "terminal"})
    assert r.message_ends_episode({"status": "finish"})
    assert not r.message_ends_episode({"status": "continue"})
    assert not r.message_ends_episode({})  # default 'continue'


@pytest.mark.parametrize("cls", [ChainRollout, StepRollout])
@pytest.mark.parametrize("status,control,expected", [
    ("success", "end", "tool_control_end"),
    ("invalid", "end", "tool_control_end"),
    ("error", "end", "tool_control_end"),
    ("invalid", None, "invalid_end"),
    ("error", None, "tool_error_end"),
    ("success", None, "tool_end"),  # Fallback for a custom policy that ends.
])
def test_tool_finish_reasons_are_shared(cls, status, control, expected):
    reason = cls.tool_finish_reason(_res(status, control))
    assert isinstance(reason, FinishReason)
    assert reason.value == expected


def test_raise_for_result():
    r = ChainRollout()
    with pytest.raises(InvalidToolCallError):
        r.raise_for_result(_res("invalid"))
    with pytest.raises(RuntimeError):
        r.raise_for_result(_res("error"))


# ---- ToolResult data model -------------------------------------------------

def test_toolresult_from_raw_control_and_status():
    r = ToolResult.from_raw(
        {"observation": "o", "status": "error", "control": "end"}, name="t", arguments={}
    )
    assert r.status == "error"      # dict status wins over the default param
    assert r.control == "end"


def test_toolresult_rejects_bad_control():
    with pytest.raises(ValueError):
        ToolResult.from_raw({"observation": "o", "control": "nope"}, name="t", arguments={})


def test_toolresult_default_control_none():
    r = ToolResult.from_raw({"observation": "o"}, name="t", arguments={})
    assert r.control is None
    assert r.status == "success"


# ---- tool-layer producers --------------------------------------------------

@pytest.mark.asyncio
async def test_submit_tool_call_unknown_tool_is_invalid_not_raised():
    # A hallucinated tool name comes back as status="invalid" (diagnosis), never raised.
    res = await submit_tool_call("does_not_exist_xyz", {}, allowed_tool_names=["a", "b"])
    assert res.status == "invalid"
    assert res.control is None
    assert "does not exist" in res.observation


@pytest.mark.asyncio
async def test_tool_body_exception_becomes_status_error():
    @tool(name="boom_tool", description="raises")
    def boom_tool(x: int):
        raise ValueError("kaboom")

    res = await submit_tool_call("boom_tool", {"x": 1}, allowed_tool_names=["boom_tool"])
    assert res.status == "error"
    assert res.control is None
    assert "kaboom" in res.observation


@pytest.mark.asyncio
async def test_tool_can_emit_control_end():
    @tool(name="done_tool", description="ends")
    def done_tool(x: int):
        return {"observation": "finished", "control": "end"}

    res = await submit_tool_call("done_tool", {"x": 1}, allowed_tool_names=["done_tool"])
    assert res.control == "end"
    assert ChainRollout().resolve_tool_control(res) == "end"
