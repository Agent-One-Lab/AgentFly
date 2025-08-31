import pytest, ray
from ....tools import code_interpreter
from ....tools.utils.rayify import rayify
from ray.util import inspect_serializability

# def test_serializability():
#     runner = rayify(code_interpreter, num_cpus=1)
#     print(inspect_serializability(runner))

# @pytest.mark.asyncio(loop_scope="session")
# async def test_rayify():
#     runner = rayify(code_interpreter, num_cpus=1)

#     ref = runner.__call__.remote(code="print('Hello, world!')", id="tid0")
#     result = await ref                       # async ray.get

#     assert result["observation"].strip() == "Hello, world!"