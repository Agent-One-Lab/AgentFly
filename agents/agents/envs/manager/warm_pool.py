from collections.abc import Callable
from typing import TypeVar, Generic
import asyncio
from ..env_base import BaseEnv
from docker.errors import NotFound, APIError

T = TypeVar("T", bound=BaseEnv)

class WarmPool(Generic[T]):
    def __init__(self, factory: Callable[[], T], size: int):
        self._factory, self._size = factory, size
        self._free: asyncio.LifoQueue[T] | None = None
        self._lock: asyncio.Lock | None = None
        self._spawn_tasks: list[asyncio.Task[None]] = []
        # self._in_pool: set[T] = set()  # Track which envs are currently in the pool
    
    @property
    def size(self):
        return self._size

    # ---------- lifecycle ----------
    async def start(self):
        # bind primitives to the *current* running loop
        self._free = asyncio.LifoQueue()
        self._lock = asyncio.Lock()
        for _ in range(self._size):
            self._spawn_tasks.append(asyncio.create_task(self._spawn()))

    async def add_envs(self, size: int):
        for _ in range(size):
            self._spawn_tasks.append(asyncio.create_task(self._spawn()))

    async def aclose(self):
        # 1. cancel all unfinished tasks
        for task in self._spawn_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._spawn_tasks, return_exceptions=True)

        # 2. close other live envs
        while not self._free.empty():
            env = await self._free.get()
            await env.aclose()
        
        # # 3. clear tracking set
        # self._in_pool.clear()

    # ---------- token operations ----------
    async def acquire(self) -> T:
        assert self._free is not None
        # print(f"[WarmPool] Current free queue size: {self._free.qsize()}")
        # print(f"[POOL] free={self._free.qsize()}/{self._size}")
        env = await self._free.get()
        # self._in_pool.discard(env)  # Remove from tracking set when acquired
        return env

    async def release(self, env: T, *, finished: bool = True):
        # Prevent the same env from being released multiple times
        # if env in self._in_pool:
        #     # print(f"[WarmPool] Warning: Env {env} already in pool, skipping release")
        #     return
            
        # We don't reset the env during release, and leave this to the user
        # if finished:
        #     await self.reset(env)
        # self._in_pool.add(env)  # Mark as in pool before putting
        await self._free.put(env)
        # print(f"[WarmPool] Current free queue size: {self._free.qsize()}")

    async def reset(self, env: T):
        try:
            await asyncio.wait_for(env.reset(), timeout=8.0)
        except Exception as e:
            try:
                await env.aclose()
            except NotFound:
                print(f"[WarmPool] Env {env} not found")
                await self._spawn()
            except APIError:
                print(f"[WarmPool] Env {env} APIError")
                await self._spawn()
            except Exception as e:
                raise e
            return

    async def _spawn(self):
        async with self._lock:
            env = self._factory()
            await env.start()
            await env.reset()
            # self._in_pool.add(env)  # Mark as in pool before putting
            await self._free.put(env)
