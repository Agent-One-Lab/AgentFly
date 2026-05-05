import pytest

from agentfly.core import Context
from agentfly.tools import code_interpreter


def test_code_schema():
    schema = code_interpreter.schema
    print(schema)


@pytest.mark.asyncio(loop_scope="session")
async def test_code_run():
    context = Context(rollout_id="demo")
    result = await code_interpreter(code='print("A print test")', context=context)
    print(result)
    await context.release_resource(scope="rollout")
    print("done")


@pytest.mark.asyncio(loop_scope="session")
async def test_code_hang():
    context = Context(rollout_id="demo")
    result = await code_interpreter(code="while True:\n  pass", context=context)
    print(result)
    await context.release_resource(scope="rollout")
    print("done")


# @pytest.mark.asyncio(loop_scope="session")
# async def test_pool_async_calls():
#     import asyncio
#
#     async def one_chain(i):
#         ctx = Context(rollout_id=f"c{i}")
#         await code_interpreter(code="x=1", context=ctx)
#         await ctx.release_resource(scope="rollout")
#
#     await asyncio.gather(*[one_chain(i) for i in range(10)])
#

@pytest.mark.asyncio(loop_scope="session")
async def test_double_release():
    context = Context(rollout_id="x")
    await code_interpreter(code="print('hi')", context=context)
    await context.release_resource(scope="rollout")
    await context.release_resource(scope="rollout")  # must return instantly (no-op)
