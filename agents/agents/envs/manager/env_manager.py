

import asyncio
import warnings
from .warm_pool import WarmPool
from ..env_base import BaseEnv


class EnvironmentManager:
    _pools: dict[type[BaseEnv], WarmPool[BaseEnv]] = {}

    # We will keep track of the acquired envs to allow request for same environment
    _acquired_envs: dict[str, BaseEnv] = {}

    @classmethod
    async def start(cls, env_cls: type[BaseEnv], size: int = 1, env_kwargs: dict | None = None):
        """
        Start a new pool for the environment class if it doesn't exist,
        or add more envs to the existing pool if the size is larger.
        If the size is smaller, do nothing.
        """
        # TODO: Currently, WarmPool will start all the envs at once. This should be fine for training, but might be wasteful for showing the demo, we may need to support feature to start a new env when acquiring, or make it a configurable option.
        key = env_cls
        if key not in cls._pools:
            cls._pools[key] = WarmPool(
                lambda: env_cls(**(env_kwargs or {})), size=size
            )
            await cls._pools[key].start()
        else:
            pool_size = cls._pools[key].size
            if pool_size < size:
                # Add more envs to the pool
                await cls._pools[key].add_envs(size - pool_size)
            else:
                pass

    @classmethod
    async def acquire(cls,
        env_cls: type[BaseEnv],
        id: str,
    ) -> BaseEnv:
        if id in cls._acquired_envs:
            return cls._acquired_envs[id]
        else:
            key = env_cls
            if key not in cls._pools:
                raise ValueError(f"Environment class {env_cls} not found. Start the environment before acquiring.")
            
            env = await cls._pools[key].acquire()
            cls._acquired_envs[id] = env
        return env

    @classmethod
    async def release(cls,
        env: BaseEnv,
        id: str,
        finished: bool = True
    ):
        key = type(env)
        if id in cls._acquired_envs:
            cls._acquired_envs.pop(id)
            await cls._pools[key].release(env, finished=finished)
        else:
            # This should be generally safe to skip
            # warnings.warn(f"Environment {id} not found during release. Skipped it.")
            pass

    @classmethod
    async def reset(cls, env: BaseEnv, env_args: dict | None = None):
        if env_args is not None:
            await env.reset(env_args)
        else:
            await env.reset()

    @classmethod
    async def aclose(cls):
        await asyncio.gather(*(p.aclose() for p in cls._pools.values()))
    
    
