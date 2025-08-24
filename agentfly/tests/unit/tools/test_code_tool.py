import asyncio
import time
from ....tools import code_interpreter
import pytest


def test_code_schema():
    schema = code_interpreter.schema
    print(schema)


@pytest.mark.asyncio(loop_scope="session")
async def test_code_run():
    result = await code_interpreter(code='print("A print test")', id='demo')
    print(result)
    await code_interpreter.release(id='demo')
    print('done')


@pytest.mark.asyncio(loop_scope="session")
async def test_code_hang():
    result = await code_interpreter(code='while True:\n  pass', id='demo')
    print(result)
    await code_interpreter.release(id='demo')
    print('done')


# @pytest.mark.asyncio(loop_scope="session")
# async def test_pool_async_calls():

#     async def one_chain(i):
#         await code_interpreter(id=f"c{i}", code="x=1")
#         await code_interpreter.release(id=f"c{i}")

#     await asyncio.gather(*[
#         one_chain(i) for i in range(code_interpreter.pool_size+5)   # over-subscribe the pool
#     ])


@pytest.mark.asyncio(loop_scope="session")
async def test_double_release():
    await code_interpreter(id="x", code="print('hi')")
    # manual double call
    await code_interpreter.release(id="x")
    await code_interpreter.release(id="x")   # must return instantly


# @pytest.mark.asyncio(loop_scope="session")
# async def test_global_clean():

#     async def one_chain(i):
#         await code_interpreter(id=f"c{i}", code="x=1")
#         # We don't release the env here, so it will be cleaned up automatically
#         # await code_interpreter.release_env(id=f"c{i}")

#     await asyncio.gather(*[
#         one_chain(i) for i in range(code_interpreter.pool_size-5)
#     ])