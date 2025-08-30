from ....tools import webshop_browser
import pytest
import asyncio

@pytest.mark.asyncio
async def test_webshop_search():
    result = await webshop_browser(action='search', value='shoes', id='testsearch')
    assert result['status'] == 'success'
    await webshop_browser.release(id='testsearch')

@pytest.mark.asyncio
async def test_webshop_search_and_next_page():
    result = await webshop_browser(action='search', value='shoes', id='testnext')
    result = await webshop_browser(action='click', value='next >', id='testnext')
    assert result['status'] == 'success'
    await webshop_browser.release(id='testnext')

@pytest.mark.asyncio
async def test_pool_async_calls():
    async def one_chain(i):
        await webshop_browser(action='search', value='shoes', id=f'test{i}')
        await webshop_browser.release(id=f'test{i}')
    await asyncio.gather(*[
        one_chain(i) for i in range(webshop_browser.pool_size+5)   # over-subscribe the pool
    ])
