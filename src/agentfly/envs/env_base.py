"""Environment base classes.

:class:`BaseEnv` is the contract every environment implements (``start``, ``reset``, ``step``,
``aclose`` / ``close``).
"""

import abc
import copy
from typing import Any


class BaseEnv(abc.ABC):
    """Minimal contract every environment must honour."""

    @abc.abstractmethod
    async def start(self) -> None:
        """
        Start the environment and allocate any resources. e.g. start a container and connect to it.
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

    # ---- branching (optional) ----
    def snapshot(self) -> Any:
        """The environment's state at this moment, for :meth:`restore` later (a rollout strategy
        branches an episode from here). The default deep-copies the instance attributes, which fits a
        pure-Python environment; an environment with external state (a container, a browser) overrides
        both methods (a filesystem or git snapshot, a checkpoint) or leaves them unsupported.
        """
        return copy.deepcopy(self.__dict__)

    def restore(self, state: Any) -> None:
        """Put the environment back into ``state`` (what :meth:`snapshot` returned)."""
        self.__dict__.clear()
        self.__dict__.update(copy.deepcopy(state))
