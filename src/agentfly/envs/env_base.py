"""Environment base classes.

:class:`BaseEnv` is the contract every environment implements (``start``, ``reset``, ``step``,
``aclose`` / ``close``).
"""

import abc
from typing import Any


class BaseEnv(abc.ABC):
    """Minimal contract every environment must honour."""

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
