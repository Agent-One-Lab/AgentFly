import pytest

from agentfly.core import Context
from agentfly.tools import webshop_browser


@pytest.mark.asyncio
async def test_webshop_search():
    ctx = Context(rollout_id="test_webshop_search")
    try:
        result = await webshop_browser(
            action="search", value="shoes", context=ctx
        )
        assert isinstance(result, str)
        assert len(result) > 0
    finally:
        await ctx.release_resource(scope="rollout")


@pytest.mark.asyncio
async def test_webshop_search_and_next_page():
    ctx = Context(rollout_id="test_webshop_next")
    try:
        result = await webshop_browser(
            action="search", value="shoes", context=ctx
        )
        assert isinstance(result, str)
        result = await webshop_browser(
            action="click", value="Next >", context=ctx
        )
        assert isinstance(result, str)
    finally:
        await ctx.release_resource(scope="rollout")
