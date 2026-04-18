"""
Ray-backed container resource.

A Ray actor owns a local enroot container on the worker node. :class:`RayContainerResource`
is an async façade so callers match :class:`ContainerResource` (``run_cmd``, lifecycle).

Other Ray workers use the container by holding an :class:`ActorHandle` and calling
``run_cmd`` via this wrapper (or ``.remote`` on the actor directly).

Driver-side ``ray.get`` calls use finite ``timeout`` values (see module constants below).
Override default cap for ``run_cmd`` when enroot ``timeout`` is omitted with environment
variable ``AGENTFLY_RAY_GET_DEFAULT_TIMEOUT_SEC`` (default ``7200``).
"""

from __future__ import annotations

import asyncio
import base64
import os
import shlex
import time
from typing import Any, Optional, TYPE_CHECKING

import ray
from enroot.errors import APIError, EnrootError, TimeoutError as EnrootTimeoutError
from ray.exceptions import ActorDiedError, GetTimeoutError as RayGetTimeoutError

from ..types import BaseResource, ContainerResourceSpec, ResourceStatus

if TYPE_CHECKING:
    from ray.actor import ActorHandle

# Slack beyond enroot/command timeout so Ray RPC can finish after the guest stops.
_RAY_GET_RUN_CMD_SLACK_SEC = 120.0
_RAY_GET_RELOAD_SEC = 300.0
_RAY_GET_NODE_OR_STATUS_SEC = 120.0
_RAY_GET_KILL_SEC = 600.0
_RAY_GET_RESOURCE_ID_SEC = 300.0
# Wait for actor __init__ (container create/start): enroot timeout + scheduling slack.
_RAY_GET_PING_SLACK_SEC = 300.0
_RAY_GET_ACTOR_DIED_MAX_ATTEMPTS = 3
_RAY_GET_ACTOR_DIED_BACKOFF_BASE_SEC = 8.0


def _default_ray_get_unbounded_cmd_sec() -> float:
    try:
        return float(os.environ.get("AGENTFLY_RAY_GET_DEFAULT_TIMEOUT_SEC", "7200"))
    except ValueError:
        return 7200.0


def _ray_get_blocking(ref: Any, timeout_sec: Optional[float]) -> Any:
    """
    ``ray.get`` with optional timeout; maps :class:`ray.exceptions.GetTimeoutError` to
    :class:`asyncio.TimeoutError`. If ``timeout_sec`` is ``None``, ``ray.get`` is unbounded.
    Retries :class:`~ray.exceptions.ActorDiedError` up to :data:`_RAY_GET_ACTOR_DIED_MAX_ATTEMPTS`
    with exponential backoff from :data:`_RAY_GET_ACTOR_DIED_BACKOFF_BASE_SEC`, then re-raises
    that error if all attempts fail.
    """
    for attempt in range(_RAY_GET_ACTOR_DIED_MAX_ATTEMPTS):
        try:
            if timeout_sec is None:
                return ray.get(ref)
            return ray.get(ref, timeout=timeout_sec)
        except RayGetTimeoutError as e:
            raise asyncio.TimeoutError(
                f"ray.get timed out after {timeout_sec} seconds"
            ) from e
        except ActorDiedError:
            if attempt + 1 >= _RAY_GET_ACTOR_DIED_MAX_ATTEMPTS:
                raise
            time.sleep(_RAY_GET_ACTOR_DIED_BACKOFF_BASE_SEC * (2**attempt))


def _require_ray_initialized() -> None:
    if not ray.is_initialized():
        raise RuntimeError(
            "This Python process is not connected to Ray. A cluster running elsewhere "
            "(e.g. `ray status` on the head) does not initialize this driver. "
            "Call ray.init(address=...) in this process, or set RAY_ADDRESS (e.g. "
            "'<head_ip>:6379') and ray.init(address='auto')."
        )


def _sync_run_cmd(
    container: Any,
    cmd: str,
    timeout: Optional[float],
    workdir: Optional[str],
    user: Optional[str],
    environment: Optional[dict],
    exec_kwargs: dict[str, Any],
) -> str:
    """Mirror :meth:`ContainerResource.run_cmd` logic synchronously (runs inside the actor)."""
    b64 = base64.b64encode(cmd.encode("utf-8")).decode("ascii")
    shell_cmd = f"echo {shlex.quote(b64)} | base64 -d | bash -s"
    exec_args: list = ["bash", "-c", shell_cmd]
    kwargs: dict = {**exec_kwargs}
    if workdir is not None:
        kwargs["workdir"] = workdir
    if user is not None:
        kwargs["user"] = user
    if environment is not None:
        kwargs["environment"] = environment
    if timeout is not None:
        kwargs["timeout"] = timeout

    try:
        result = container.exec_run(exec_args, **kwargs)
    except EnrootTimeoutError as e:
        if timeout is not None:
            raise asyncio.TimeoutError(
                f"Command timed out after {timeout} seconds: {cmd}"
            ) from e
        raise asyncio.TimeoutError(str(e)) from e

    output = result.output if hasattr(result, "output") else str(result)
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output if output is not None else ""


class _RayEnrootContainerActorBase:
    """
    Owns one enroot container on the Ray worker process (local placement only).

    Wrapped with ``@ray.remote`` as :data:`RayEnrootContainerActor`.
    """

    def __init__(
        self,
        resource_id: str,
        spec: ContainerResourceSpec,
        start_timeout: Optional[float] = 1800.0,
    ):
        from enroot import from_env, random_name

        if spec.category != "container":
            raise ValueError(
                f"RayEnrootContainerActor supports only category 'container', got {spec.category!r}"
            )
        name = resource_id or random_name(prefix="res")
        image = spec.image or "ubuntu:22.04"
        timeout = start_timeout
        create_kwargs: dict[str, Any] = {
            "name": name,
            "environment": spec.environment or {},
            "mount": spec.mount or {},
            "ports": spec.ports,
            "timeout": timeout,
        }
        start_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "environment": spec.environment or {},
            "mount": spec.mount or {},
        }
        client = from_env()
        try:
            self._container = client.containers.create(image, **create_kwargs)
            self._container.start(**start_kwargs)
        except (TimeoutError, EnrootTimeoutError) as e:
            raise asyncio.TimeoutError(
                f"Container create/start timed out for image={image!r}, name={name!r}, timeout={timeout!r}."
            ) from e
        except APIError as e:
            raise RuntimeError(
                f"Container create/start failed for image={image!r}, name={name!r}: {e}"
            ) from e
        except EnrootError as e:
            raise RuntimeError(
                f"Container create/start failed for image={image!r}, name={name!r}: {e}"
            ) from e

        self._resource_id = self._container.name

    def ping(self) -> str:
        """Cheap call to wait for actor ``__init__`` to finish when using a fresh handle."""
        return "ok"

    def get_resource_id(self) -> str:
        return self._resource_id

    def get_ray_node_id(self) -> str:
        """Ray node id for the worker process hosting this actor (for placement tests)."""
        return ray.get_runtime_context().get_node_id()

    async def reload(self) -> None:
        self._container.reload()

    async def get_status_str(self) -> str:
        self._container.reload()
        return str(getattr(self._container, "status", "exited"))

    async def kill_container(self) -> None:
        if getattr(self, "_container", None) is not None:
            if hasattr(self._container, "kill_async"):
                await self._container.kill_async()
            else:
                self._container.kill()
            self._container = None

    async def raw_exec_run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward Docker-SDK-like ``exec_run`` to enroot, using the async enroot path.

        If the caller wrapped the *guest* command with GNU ``timeout ...`` but did not
        pass ``timeout=...`` to exec_run, we infer a reasonable host-side timeout and
        pass it to enroot-py so the host-side ``enroot exec`` cannot wedge.
        """
        if getattr(self, "_container", None) is None:
            raise RuntimeError("Container already stopped")

        # Infer host timeout from a leading `timeout <secs> ...` wrapper in cmd.
        # This strengthens behavior when callers embed GNU `timeout` inside cmd but
        # don't set enroot-py's `timeout=` (host-level guard).
        if kwargs.get("timeout", None) is None:
            cmd_obj = None
            if args:
                cmd_obj = args[0]
            elif "cmd" in kwargs:
                cmd_obj = kwargs.get("cmd")

            host_timeout: Optional[float] = None
            if isinstance(cmd_obj, list) and len(cmd_obj) >= 2 and str(cmd_obj[0]) == "timeout":
                try:
                    host_timeout = float(str(cmd_obj[1]))
                except Exception:
                    host_timeout = None
            elif isinstance(cmd_obj, str):
                try:
                    parts = shlex.split(cmd_obj)
                    if len(parts) >= 2 and parts[0] == "timeout":
                        host_timeout = float(parts[1])
                except Exception:
                    host_timeout = None

            if host_timeout is not None:
                kwargs["timeout"] = host_timeout

        try:
            if hasattr(self._container, "exec_run_async"):
                return await self._container.exec_run_async(*args, **kwargs)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._container.exec_run, *args, **kwargs
            )
        except EnrootTimeoutError as e:
            raise asyncio.TimeoutError(str(e)) from e

    async def run_cmd(
        self,
        cmd: str,
        timeout: Optional[float] = None,
        workdir: Optional[str] = None,
        user: Optional[str] = None,
        environment: Optional[dict] = None,
        exec_extra: Optional[dict[str, Any]] = None,
    ) -> str:
        b64 = base64.b64encode(cmd.encode("utf-8")).decode("ascii")
        shell_cmd = f"echo {shlex.quote(b64)} | base64 -d | bash -s"
        exec_args: list = ["bash", "-c", shell_cmd]

        kwargs: dict[str, Any] = dict(exec_extra or {})
        if workdir is not None:
            kwargs["workdir"] = workdir
        if user is not None:
            kwargs["user"] = user
        if environment is not None:
            kwargs["environment"] = environment
        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            if hasattr(self._container, "exec_run_async"):
                result = await self._container.exec_run_async(exec_args, **kwargs)
            else:
                # Fallback: shouldn't happen if enroot-py provides exec_run_async.
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, self._container.exec_run, exec_args, **kwargs
                )
        except EnrootTimeoutError as e:
            if timeout is not None:
                raise asyncio.TimeoutError(
                    f"Command timed out after {timeout} seconds: {cmd}"
                ) from e
            raise asyncio.TimeoutError(str(e)) from e

        output = result.output if hasattr(result, "output") else str(result)
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output if output is not None else ""


RayEnrootContainerActor = ray.remote(num_cpus=1)(_RayEnrootContainerActorBase)


class RaySyncExecRunProxy:
    """
    Blocking ``exec_run`` facade for Docker-SDK-shaped call sites (e.g. R2E ``reward_from_container``).

    Used from threads via ``asyncio.to_thread``; each call ``ray.get``s the actor RPC.
    """

    __slots__ = ("_actor",)

    def __init__(self, actor: Any) -> None:
        self._actor = actor

    def exec_run(self, *args: Any, **kwargs: Any) -> Any:
        ref = self._actor.raw_exec_run.remote(*args, **kwargs)
        kw_t = kwargs.get("timeout")
        if kw_t is not None:
            get_t = float(kw_t) + _RAY_GET_RUN_CMD_SLACK_SEC
        else:
            get_t = _default_ray_get_unbounded_cmd_sec()
        return _ray_get_blocking(ref, get_t)


class RayContainerResource(BaseResource):
    """
    Ray-backed implementation of :class:`BaseResource` for container workloads.

    This wrapper proxies lifecycle and command execution to a remote actor that
    owns an enroot container on a Ray worker node. Its public behavior mirrors
    :class:`ContainerResource` so tool/reward code can use the same interface
    across local and Ray backends.

    Note:
        The `container` property returns a Ray `ActorHandle`, not a direct
        enroot container object.
    """

    def __init__(self, actor: "ActorHandle", resource_id: str, spec: ContainerResourceSpec):
        """
        Initialize a Ray-backed container resource.

        Args:
            actor: Ray actor handle owning the underlying container lifecycle.
            resource_id: Stable id used by the resource engine for tracking/reuse.
            spec: Resource specification that produced this resource.
        """
        self._actor = actor
        self._resource_id = resource_id
        self._spec = spec

    @property
    def resource_id(self) -> str:
        return self._resource_id

    @property
    def category(self) -> str:
        return "container"

    @property
    def container(self) -> Any:
        """Ray actor handle; remote execution uses :meth:`run_cmd` or ``actor.run_cmd.remote``."""
        return self._actor

    def sync_exec_run_proxy(self) -> RaySyncExecRunProxy:
        """Object with blocking ``exec_run`` matching enroot/docker SDK for sync reward/eval code."""
        return RaySyncExecRunProxy(self._actor)

    async def _ray_get(self, ref: Any, *, timeout_sec: Optional[float]) -> Any:
        """Resolve a Ray object ref in a worker thread with timeout/error mapping."""
        return await asyncio.to_thread(_ray_get_blocking, ref, timeout_sec)

    async def start(self) -> None:
        """Verify actor/container availability by reloading container state remotely."""
        await self._ray_get(self._actor.reload.remote(), timeout_sec=_RAY_GET_RELOAD_SEC)

    async def get_ray_node_id(self) -> str:
        """Return the Ray ``NodeID`` where this resource's actor is running."""
        return await self._ray_get(
            self._actor.get_ray_node_id.remote(),
            timeout_sec=_RAY_GET_NODE_OR_STATUS_SEC,
        )

    async def get_status(self) -> ResourceStatus:
        """Return lifecycle status mapped from the remote container status string."""
        s = await self._ray_get(
            self._actor.get_status_str.remote(),
            timeout_sec=_RAY_GET_NODE_OR_STATUS_SEC,
        )
        if s == "running":
            return ResourceStatus.RUNNING
        if s == "exited":
            return ResourceStatus.STOPPED
        return ResourceStatus.PENDING

    async def control(self, **kwargs: Any) -> None:
        """Update runtime limits if supported (currently a no-op for Ray container actor)."""
        pass

    async def close(self) -> None:
        """Alias for :meth:`end`."""
        await self.end()

    async def reset(self) -> None:
        """Reset is not supported for this resource type."""
        raise NotImplementedError("RayContainerResource does not support reset.")

    async def end(self) -> None:
        """Kill the remote container, then terminate the backing Ray actor."""
        try:
            await self._ray_get(
                self._actor.kill_container.remote(),
                timeout_sec=_RAY_GET_KILL_SEC,
            )
        finally:
            try:
                ray.kill(self._actor, no_restart=True)
            except Exception:
                pass

    async def run_cmd(
        self,
        cmd: str,
        timeout: Optional[float] = None,
        *,
        workdir: Optional[str] = None,
        user: Optional[str] = None,
        environment: Optional[dict] = None,
        **exec_kwargs: Any,
    ) -> str:
        """
        Execute a shell command in the container on the actor's node.

        Same base64/bash layering as :meth:`ContainerResource.run_cmd`.
        """
        ref = self._actor.run_cmd.remote(
            cmd,
            timeout,
            workdir,
            user,
            environment,
            exec_kwargs,
        )
        if timeout is not None:
            get_timeout = float(timeout) + _RAY_GET_RUN_CMD_SLACK_SEC
        else:
            get_timeout = _default_ray_get_unbounded_cmd_sec()
        out = await self._ray_get(ref, timeout_sec=get_timeout)
        if isinstance(out, str):
            return out
        return str(out)


async def create_ray_container_resource(
    spec: ContainerResourceSpec,
    resource_id: str,
    *,
    start_timeout: Optional[float] = 1800.0,
    ray_actor_options: Optional[dict[str, Any]] = None,
) -> RayContainerResource:
    """
    Create a Ray actor on the cluster (local enroot on that worker) and wrap it.

    Requires ``ray.init(...)`` and enroot on the worker that runs the actor.
    Use ``ray_actor_options`` (e.g. ``num_cpus``, ``num_gpus``, ``resources``,
    ``scheduling_strategy``) to pin the container to a specific node.
    """
    _require_ray_initialized()
    if spec.category != "container":
        raise ValueError(
            f"create_ray_container_resource supports only category 'container', got {spec.category!r}"
        )
    opts = dict(ray_actor_options or {})
    st = start_timeout if start_timeout is not None else 1800.0
    actor = RayEnrootContainerActor.options(**opts).remote(
        resource_id,
        spec,
        start_timeout,
    )
    ping_timeout = float(st) + _RAY_GET_PING_SLACK_SEC
    await asyncio.to_thread(_ray_get_blocking, actor.ping.remote(), ping_timeout)
    rid = await asyncio.to_thread(
        _ray_get_blocking,
        actor.get_resource_id.remote(),
        _RAY_GET_RESOURCE_ID_SEC,
    )
    res = RayContainerResource(actor, rid, spec)
    await res.start()
    return res
