import pytest

from agentfly.core import Context
from agentfly.tools import code_interpreter


def test_code_schema():
    schema = code_interpreter.schema
    assert schema["function"]["name"] == "code_interpreter"
    assert schema["function"]["parameters"]["required"] == ["code"]


@pytest.mark.asyncio(loop_scope="session")
async def test_code_run():
    context = Context(rollout_id="demo")
    result = await code_interpreter(code='print("A print test")', context=context)
    await context.end_resource(scope="global")
    assert result.observation.strip() == "A print test", result.observation


@pytest.mark.asyncio(loop_scope="session")
async def test_code_sees_top_level_names():
    # Functions must see top-level functions and imports, as in a normal script.
    context = Context(rollout_id="demo")
    code = "import math\ndef fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\nprint(fib(10), round(math.pi, 2))"
    result = await code_interpreter(code=code, context=context)
    await context.end_resource(scope="global")
    assert result.observation.strip() == "55 3.14", result.observation


@pytest.mark.asyncio(loop_scope="session")
async def test_code_hang():
    context = Context(rollout_id="demo")
    result = await code_interpreter(code="while True:\n  pass", context=context)
    await context.end_resource(scope="global")
    # The CPU-time limit (8 s) fires before the 10 s wall-clock limit.
    assert "limit exceeded" in result.observation or "timed out" in result.observation.lower(), result.observation


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
    # release is a no-op for the rollout scope (code_interpreter acquires
    # scope="global"); calling it twice must simply not raise.
    await context.release_resource(scope="rollout")
    await context.release_resource(scope="rollout")  # must return instantly (no-op)
    # actually tear the global sandbox down so no container is leaked.
    await context.end_resource(scope="global")
