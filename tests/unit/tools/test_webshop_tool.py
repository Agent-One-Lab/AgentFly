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
        print(result)
        assert isinstance(result, dict)
    finally:
        await ctx.release_resource(scope="rollout")


@pytest.mark.asyncio
async def test_webshop_search_and_next_page():
    ctx = Context(rollout_id="test_webshop_next")
    try:
        result = await webshop_browser(
            action="search", value="shoes", context=ctx
        )
        print(result)
        assert isinstance(result, dict)
        result = await webshop_browser(
            action="click", value="Next >", context=ctx
        )
        print(result)
        assert isinstance(result, dict)
    finally:
        await ctx.release_resource(scope="rollout")
