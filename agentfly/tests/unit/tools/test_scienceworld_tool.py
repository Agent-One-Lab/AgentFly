from ....tools import scienceworld_explorer
import pytest
import asyncio

@pytest.mark.asyncio
async def test_science_world_explorer():
    result = await scienceworld_explorer(action='look around', id='testlook')
    assert result['status'] == 'success'
    await scienceworld_explorer.release(id='testlook')

# @pytest.mark.asyncio
# async def test_pool_async_calls():
#     async def one_chain(i):
#         await scienceworld_explorer(action='look around', id=f'test{i}')
#         await scienceworld_explorer.release(id=f'test{i}')
#     await asyncio.gather(*[
#         one_chain(i) for i in range(scienceworld_explorer.pool_size+5)   # over-subscribe the pool
#     ])
