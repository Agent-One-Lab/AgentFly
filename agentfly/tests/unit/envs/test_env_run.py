import asyncio
import time
from ....envs.manager.warm_pool import WarmPool
from ....envs.python_env import PythonSandboxEnv
import pytest
import requests


# @pytest.fixture
# def test_server():
#     port = 8000
#     url = f"http://localhost:{port}/exec"
#     response = requests.post(url, json={"code": "print('Hello, world!')"})
#     assert response.status_code == 200
#     assert response.json() == {"output": "Hello, world!\n"}, f"Response: {response.json()}"


@pytest.mark.asyncio
async def test_python_sandbox_env():
    env = PythonSandboxEnv()
    assert env is not None
    await env.start()
    await env.reset()
    obs = await env.step("print('Hello, world!')")
    assert obs == "Hello, world!\n", f"Response: {obs}"
    await env.aclose()



N_REQUESTS = 1_00

@pytest.mark.asyncio
async def test_python_sandbox_env_concurrent_requests():
    # Record time
    start_time = time.time()
    env = PythonSandboxEnv(max_episodes=2)   # only one reset needed
    await env.start()
    await env.reset()

    async def run(i: int):
        code = f"print({i})"
        obs = await env.step(code)
        return i, obs.strip()

    tasks = [asyncio.create_task(run(i)) for i in range(N_REQUESTS)]
    results = await asyncio.gather(*tasks)

    for i, out in results:
        assert out == str(i)

    await env.aclose()

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

# N_ENVS       = 1000     # total environments you want to exercise
# MAX_PARALLEL = 32    # how many containers may run at the same time

# @pytest.mark.asyncio
# async def test_python_sandbox_env_many_instances():
#     """
#     Launch `N_ENVS` separate PythonSandboxEnv instances, each in its own Docker
#     container, run one tiny snippet, and close them again.

#     Concurrency is capped with an `asyncio.Semaphore` so that the host isn't
#     flooded with 1 000 simultaneous containers.
#     """
#     sem = asyncio.Semaphore(MAX_PARALLEL)
#     errors = []
#     start_time = time.time()
#     async def run_single(i: int):
#         # limit fan-out
#         async with sem:
#             env = PythonSandboxEnv()          # brand-new container
#             try:
#                 await env.start()
#                 await env.reset()
#                 v = random.randint(1, 999)    # different code per env
#                 obs = await env.step(f"print({v})")
#                 # ----- assertions -------------------------------------------
#                 assert obs.strip() == str(v), f"id={i}: wrong output {obs!r}"
#             except Exception as exc:          # collect failures but keep going
#                 errors.append(exc)
#             finally:
#                 await env.close()

#     # launch all tasks concurrently (respecting the semaphore)
#     await asyncio.gather(*(run_single(i) for i in range(N_ENVS)))

#     # bubble up any collected failures so pytest marks the test as failed
#     if errors:
#         raise AssertionError(f"{len(errors)} failures: {errors[:3]}…")
#     print(f"Time taken: {time.time() - start_time} seconds")


# @pytest.mark.asyncio
# async def test_python_sandbox_env_many_instances_pool():
#     """
#     Launch `N_ENVS` separate PythonSandboxEnv instances, each in its own Docker
#     container, run one tiny snippet, and close them again.

#     Concurrency is capped with an `asyncio.Semaphore` so that the host isn't
#     flooded with 1 000 simultaneous containers.
#     """
#     sem = asyncio.Semaphore(MAX_PARALLEL)
#     errors = []
#     start_time = time.time()
#     pool = WarmPool(lambda: PythonSandboxEnv(), size=16)
#     await pool.start()
#     async def run_single(i: int):
#         # limit fan-out
#         async with sem:
#             try:
#                 v = random.randint(1, 999)    # different code per env
#                 env = await pool.acquire()
#                 obs = await env.step(f"print({v})")
#                 # ----- assertions -------------------------------------------
#                 assert obs.strip() == str(v), f"id={i}: wrong output {obs!r}"

#             except Exception as exc:          # collect failures but keep going
#                 errors.append(exc)
#             finally:
#                 await pool.release(env)

#     # launch all tasks concurrently (respecting the semaphore)
#     await asyncio.gather(*(run_single(i) for i in range(N_ENVS)))

#     # bubble up any collected failures so pytest marks the test as failed
#     if errors:
#         raise AssertionError(f"{len(errors)} failures: {errors[:3]}…")
#     print(f"Time taken: {time.time() - start_time} seconds")
#     await pool.close()