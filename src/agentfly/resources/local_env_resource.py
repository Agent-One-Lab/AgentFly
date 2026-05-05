"""
Local in-process env as a resource.

:class:`LocalEnvResource` adapts a :class:`~agentfly.envs.env_base.BaseEnv`
instance to the :class:`BaseResource` contract so it can flow through
:class:`ResourceEngine` like containerized envs do. Attribute access on the
wrapper transparently delegates to the underlying env, so consumers
(tools, rewards) can interact with the env directly.
"""

from __future__ import annotations

from typing import Any

from .types import BaseResource, LocalEnvResourceSpec, ResourceStatus


class LocalEnvResource(BaseResource):
    """Wrap an in-process :class:`BaseEnv` so it satisfies :class:`BaseResource`.

    Lifecycle methods (``start``, ``reset``, ``end``, ``close``) forward to
    the wrapped env. ``get_status`` always reports ``RUNNING`` because there
    is no external process to inspect, and ``control`` is a no-op. Any other
    attribute access (``env.step(...)``, ``env.some_state``, ...) is
    delegated via ``__getattr__``.
    """

    def __init__(self, env: Any, resource_id: str, spec: LocalEnvResourceSpec):
        # Use object.__setattr__ to bypass our own __getattr__ during init.
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_resource_id", resource_id)
        object.__setattr__(self, "_spec", spec)

    @property
    def resource_id(self) -> str:
        return self._resource_id

    @property
    def category(self) -> str:
        return "local_env"

    @property
    def env(self) -> Any:
        """The underlying :class:`BaseEnv` instance."""
        return self._env

    async def start(self) -> None:
        await self._env.start()

    async def reset(self, *args: Any, **kwargs: Any) -> Any:
        return await self._env.reset(*args, **kwargs)

    async def get_status(self) -> ResourceStatus:
        return ResourceStatus.RUNNING

    async def control(self, **kwargs: Any) -> None:
        pass

    async def end(self) -> None:
        await self._env.aclose()

    async def close(self) -> None:
        await self._env.aclose()

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires for attributes not found via normal lookup,
        # so it never shadows the BaseResource contract above.
        env = object.__getattribute__(self, "_env")
        return getattr(env, name)
