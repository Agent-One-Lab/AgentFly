""":class:`WarmPool`: a pool of pre-started environments created by a factory, handed out and returned (reset) for reuse."""

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar

from docker.errors import APIError, NotFound

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..env_base import BaseEnv

T = TypeVar("T", bound="BaseEnv")


class WarmPool(Generic[T]):
    def __init__(self, factory: Callable[[], T], size: int):
        self._factory, self._size = factory, size
        self._free: asyncio.LifoQueue[T] | None = None
        self._lock: asyncio.Lock | None = None
        self._spawn_tasks: list[asyncio.Task[None]] = []

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

    # ---------- token operations ----------
    async def acquire(self) -> T:
        assert self._free is not None
        env = await self._free.get()
        return env

    async def release(self, env: T, *, finished: bool = True):
        # We don't reset the env during release, and leave this to the user
        await self._free.put(env)

    async def reset(self, env: T):
        try:
            await asyncio.wait_for(env.reset(), timeout=8.0)
        except Exception:
            try:
                await env.aclose()
            except NotFound:
                logger.warning(f"[WarmPool] Env {env} not found")
                await self._spawn()
            except APIError:
                logger.warning(f"[WarmPool] Env {env} APIError")
                await self._spawn()
            except Exception as e:
                raise e
            return

    async def _spawn(self):
        async with self._lock:
            env = self._factory()
            await env.start()
            await env.reset()
            await self._free.put(env)
