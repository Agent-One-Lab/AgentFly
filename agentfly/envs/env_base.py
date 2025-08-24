# env_base.py
import abc, asyncio
import time
from typing import Any, Mapping
import warnings
# import docker
from .manager import enroot
from .manager.resource import GLOBAL_ENVS

class BaseEnv(abc.ABC):
    """Minimal contract every environment must honour."""
    def __init__(self):
        GLOBAL_ENVS.add(self)

    @abc.abstractmethod
    async def start(self) -> None:
        """
        Start the environment and allocate any resources. e.g. start a docker container and connect it.
        This method is generally slow, and is often called when the environment is first created, the environment is broken or lost.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def reset(self) -> Any:
        """
        Reset the environment to its initial state. e.g. remove any created files, reset settings in operating system.
        Compared to start, this method is generally faster, and used when we want to reuse the environment for lower latency.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def step(self, action: str) -> str:
        """
        Take an action in the environment and return the observation.
        This method is the interface that the Tool will call. A tool give the action to the environment, and the environment will return the observation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def aclose(self) -> None:
        """
        Release everything allocated by the environment.
        This method is called when the environment is no longer needed, or the environment is broken.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Release everything allocated by the environment.
        This is the synchronous version of aclose.
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    async def acquire():
        """Create a new environment."""
        raise NotImplementedError


class SupportsDocker:
    """Adds Docker lifecycle; mix into envs that want it."""
    _container = None     # set by subclass
    async def _docker_start(self, image: str, runtime: str = "runc", **run_kw):
        """
        Launch the container without blocking the asyncio event-loop.
        """
        # client = docker.from_env()
        client = enroot.from_env()
        loop   = asyncio.get_running_loop()

        # run docker-py call in default ThreadPoolExecutor
        retry_times = 3
        for _ in range(retry_times):
            self._container = await asyncio.to_thread(
                client.containers.run,
                image,
                detach=True,
                auto_remove=True,
                runtime=runtime,
                **run_kw,
            )

            # wait until the container is actually running (non-blocking poll)
            deadline = time.time() + 60
            while self._container.status != "running":
                await asyncio.sleep(0.1)
                await asyncio.to_thread(self._container.reload)
                if time.time() > deadline:
                    # raise RuntimeError("container failed to start")
                    warnings.warn("container failed to start, retrying...")
                    continue
            break
        

    async def _docker_stop(self):
        if self._container:
            await asyncio.to_thread(self._container.kill)
            self._container = None

    def get_container_logs(self) -> str:
        """Get logs from the container if it exists."""
        if self._container:
            try:
                return self._container.logs().decode('utf-8')
            except Exception as e:
                return f"Could not get container logs: {str(e)}"
        return "Container not found"
