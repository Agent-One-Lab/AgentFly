import pytest

from agentfly.core import Context
from agentfly.tools import webshop_browser
from agentfly.tools.types import ToolResult


@pytest.mark.asyncio(loop_scope="session")
async def test_webshop_search():
    ctx = Context(rollout_id="test_webshop_search")
    try:
        result = await webshop_browser(
            action="search", value="shoes", context=ctx
        )
        print(result)
        assert isinstance(result, ToolResult) and result.status == "success", result
    finally:
        await ctx.end_resource(scope="global")


@pytest.mark.asyncio(loop_scope="session")
async def test_webshop_search_and_next_page():
    ctx = Context(rollout_id="test_webshop_next")
    try:
        result = await webshop_browser(
            action="search", value="shoes", context=ctx
        )
        print(result)
        assert isinstance(result, ToolResult) and result.status == "success", result
        result = await webshop_browser(
            action="click", value="Next >", context=ctx
        )
        print(result)
        assert isinstance(result, ToolResult) and result.status == "success", result
    finally:
        await ctx.end_resource(scope="global")
