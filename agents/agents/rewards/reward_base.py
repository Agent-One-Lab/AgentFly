from abc import ABC
import asyncio
from typing import List, Optional
from ..envs.manager.env_manager import EnvironmentManager
from ..envs.env_base import BaseEnv

# Global reward registry
REWARD_REGISTRY = {}


class RewardFunction(ABC):
    """
    Base class for reward functions
    Currently, we require all reward functions to have two arguments: prediction and golden answer, and return a float number.
    """
    
    def __init__(self, name=None, func=None, env_cls: type[BaseEnv] | None = None, pool_size: int = -1, env_kwargs: dict | None = None):
        self.name = name
        self.func = func
        self.keys = None
        self.env_cls = env_cls
        self.pool_size = pool_size
        self.env_kwargs = env_kwargs
        self._envs: dict[str, BaseEnv] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.initialized = False
        # Get function signature for filtering kwargs
        if func is not None:
            import inspect
            self._func_sig = inspect.signature(func)
    
    def _filter_kwargs(self, kwargs: dict) -> dict:
        """Filter kwargs to only include parameters that the function accepts."""
        if not hasattr(self, '_func_sig'):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in self._func_sig.parameters}
    
    async def __call__(self, prediction: str, **kwargs) -> float:
        """
        Allow rewards to be called directly as functions
        Returns:
            reward_value_or_dict: A float number or a dictionary with the following keys:
                - "reward": A float number
                - "any other keys": All must be a float number
        """
        if not self.initialized:
            await EnvironmentManager.start(self.env_cls, size=self.pool_size, env_kwargs=self.env_kwargs)
            self.initialized = True

        if self.env_cls is None:
            filtered_kwargs = self._filter_kwargs(kwargs)
            if asyncio.iscoroutinefunction(self.func):
                reward_value_or_dict = await self.func(prediction, **filtered_kwargs)
            else:
                reward_value_or_dict = self.func(prediction, **filtered_kwargs)
        else:
            id = kwargs.pop('id', None)
            if id is None:
                raise ValueError("id is required for rewards with environments.")
            if id not in self._envs:
                self._envs[id] = await EnvironmentManager.acquire(self.env_cls, id=id)
                self._locks[id] = asyncio.Lock()
            async with self._locks[id]:
                assert kwargs.get("env", None) is None, "env is not allowed to be passed to rewards with environments."
                try:
                    filtered_kwargs = self._filter_kwargs(kwargs)
                    if asyncio.iscoroutinefunction(self.func):
                        reward_value_or_dict = await self.func(env=self._envs[id], prediction=prediction, **filtered_kwargs)
                    else:
                        reward_value_or_dict = self.func(env=self._envs[id], prediction=prediction, **filtered_kwargs)
                finally:
                    pass
        
        if isinstance(reward_value_or_dict, dict):
            # TODO: Check if the keys are the same for all calls?
            if self.keys is None:
                self.keys = reward_value_or_dict.keys()
            return reward_value_or_dict
        elif isinstance(reward_value_or_dict, float):
            return {"reward": reward_value_or_dict}
        else:
            raise ValueError(f"Invalid reward: {reward_value_or_dict}, must be a float number or a dictionary with the following \"reward\" as a key.")

    async def release(self, id: str, success: bool = True):
        """Release a specific environment."""
        if not self.env_cls or id not in self._envs:
            return
        
        env = self._envs.pop(id)
        self._locks.pop(id)
        await EnvironmentManager.release(env, id=id, finished=success)

    def __repr__(self):
        return f"RewardFunction(name={self.name})"


def reward(
    name: Optional[str] = None,
    env_cls: type[BaseEnv] | None = None,
    env_kwargs: dict | None = None,
    pool_size: int = -1,
    auto_register: bool = True
):
    """
    Decorator that creates a RewardFunction and registers it.
    
    Similar to the @tool decorator in tool system.
    
    Args:
        name: The name of the reward (defaults to function name)
        llm_config: Configuration for an LLM backend (if this reward uses an LLM judge)
        auto_register: Whether to automatically register in REWARD_REGISTRY
        
    Returns:
        A RewardFunction instance
        
    Raises:
        ValueError: If the decorated function does not have exactly two parameters: prediction and golden_answer
                    or three parameters: prediction, golden_answer, llm_backend (when llm_config is provided)
    """
    def decorator(func):
        # Validate function parameters
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        
        expected_params = ['prediction']
        if env_cls is not None:
            expected_params.append('env')
            
        param_names = [p.name for p in params]
        for expected_param in expected_params:
            if expected_param not in param_names:
                raise ValueError(f"Reward function {func.__name__} must have parameter named {expected_param}")
        
        func_name = func.__name__
        if func_name and name:
            final_name = name
        else:
            final_name = func_name
        
        # Create an instance with the proper name
        reward_instance = RewardFunction(name=final_name, func=func, env_cls=env_cls, pool_size=pool_size, env_kwargs=env_kwargs)
        
        # Register the reward if auto_register is True
        if auto_register:
            register_reward(final_name, reward_instance)
            
        return reward_instance
    return decorator


def register_reward(reward_name: str, reward_function: RewardFunction) -> None:
    """
    Register a reward in the registry.
    
    Args:
        reward_name: The name of the reward
        reward_function: The reward function instance
    """
    global REWARD_REGISTRY
    REWARD_REGISTRY[reward_name.lower()] = reward_function


def get_reward_from_name(reward_name: str) -> RewardFunction:
    """
    Get a reward function by name.
    
    Args:
        reward_name: Name of the reward function
            
    Returns:
        A RewardFunction instance
        
    Raises:
        KeyError: If the reward name is not found in the registry
    """
    global REWARD_REGISTRY
    reward_name = reward_name.lower()
    if reward_name not in REWARD_REGISTRY:
        raise KeyError(f"Unknown reward: '{reward_name}'. Available rewards: {list(REWARD_REGISTRY.keys())}")
    return REWARD_REGISTRY[reward_name]


def get_rewards_from_names(reward_names: List[str]) -> List[RewardFunction]:
    """
    Get multiple reward functions by name.
    
    Args:
        reward_names: List of reward names
        
    Returns:
        List of RewardFunction instances
    """
    global REWARD_REGISTRY
    return [get_reward_from_name(name) for name in reward_names]


def list_available_rewards() -> List[str]:
    """
    List all available rewards.
    
    Returns:
        List of reward names
    """
    global REWARD_REGISTRY
    return list(REWARD_REGISTRY.keys())


@reward(name="fake_reward")
def fake_reward(prediction: str, **kwargs) -> float:
    return 0.0