import asyncio
from ....envs.redis_env import RedisEnv
import pytest

# @pytest.mark.asyncio
# async def test_redis_env_acquire():
#     env = await RedisEnv.acquire()
#     assert env is not None

# @pytest.mark.asyncio
# async def test_env_run():
#     env = await RedisEnv.acquire()
#     assert env is not None
#     obs = await env.step("Donald Trump")
#     assert obs == """1. Donald Trump - Wikipedia Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States.\n2. President Donald J. Trump - The White House President Donald J. Trump is returning to the White House to build upon his previous successes and use his mandate to reject the extremist policies.\n3. President Donald J. Trump (@realdonaldtrump) - Instagram 34M Followers, 47 Following, 7482 Posts - President Donald J. Trump (@realdonaldtrump) on Instagram: "45th & 47th President of the United States\"""", f"Got {obs}"


# @pytest.mark.asyncio
# async def test_env_async_calls():
#     env = RedisEnv()
#     await env.start()
#     await env.reset()
#     search_queries = [
#         "Donald Trump",
#         "Best boxer in the world",
#         "Best football player in the world",
#     ]
#     results = await asyncio.gather(*[env.step(query) for query in search_queries])
#     assert len(results) == len(search_queries)
#     for i in range(len(results)):
#         print(results[i])
#     await env.aclose()

