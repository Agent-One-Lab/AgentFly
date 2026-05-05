import inspect
from typing import Callable, List, Optional

from .types import RewardResult

# Global reward registry
REWARD_REGISTRY = {}


async def calculate_reward(reward_fn: Callable | None, **kwargs) -> Optional[RewardResult]:
    """Execute a reward function and normalize its return into a ``RewardResult``.

    The reward function may return a ``float``, a ``dict`` with a ``reward``
    key (plus arbitrary extras), a ``RewardResult``, or ``None``.
    All forms are funneled through :meth:`RewardResult.from_raw`.

    Returns ``None`` only when ``reward_fn`` itself is ``None`` (i.e. the agent
    has no reward configured). When a reward function is set, this always
    returns a :class:`RewardResult`.
    """
    if reward_fn is None:
        return None

    raw = reward_fn(**kwargs)
    if inspect.isawaitable(raw):
        raw = await raw
    return RewardResult.from_raw(raw)


class BaseReward:
    """
    Base class for reward functions.

    Supports both decorator-based and inheritance-based definitions.
    All arguments must be passed as keyword arguments (dictionary args), similar to tools.

    1. Decorator-based (existing pattern):

    ```python
    @reward(name="my_reward")
    def my_reward_function(prediction: str, answer: str) -> float:
        return 1.0 if prediction == answer else 0.0

    # Call with keyword arguments
    result = await my_reward(prediction="hello", answer="hello")
    ```

    2. Inheritance-based (new pattern):

    ```python
    class MyReward(BaseReward):
        # Class-level metadata (shared across all instances)
        name = "my_reward"

        def __init__(self, threshold: float = 0.5):
            super().__init__()  # Class attributes are set at class definition time
            self.threshold = threshold  # Instance data

        def call(self, prediction: str, answer: str) -> float:
            \"\"\"
            Calculate reward based on prediction and answer.

            Args:
                prediction (str): The predicted answer
                answer (str): The correct answer

            Returns:
                float: The reward value
            \"\"\"
            return 1.0 if prediction == answer else 0.0

    # Create instance and call with keyword arguments
    my_reward = MyReward()
    result = await my_reward(prediction="hello", answer="hello")
    ```

    Note: Metadata (name, etc.) is stored as class attributes.
    Stateful rewards get resources via Context (injected by caller): use
    context.acquire_resource(spec=..., scope="rollout", backend="local").
    """

    # ========== Class Attributes ==========
    name: str | None = None
    auto_register: bool = True

    # ========== Instance Attributes ==========
    def __init__(
        self,
        name: str | None = None,
        func: Callable | None = None,
        auto_register: bool = True,
    ):
        """
        Initialize a reward function instance.

        Args:
            name: Optional name override (for decorator-based rewards)
            func: Optional function (for decorator-based rewards)
            auto_register: Whether to automatically register this reward instance (defaults to True)

        Note:
            - For stateful rewards, accept context: Context and call
              context.acquire_resource(spec=..., scope="rollout", backend="local").
            - Caller must pass context (and any env) in kwargs when invoking the reward.
        """
        cls = type(self)

        if name is not None:
            cls.name = name
        if "auto_register" not in cls.__dict__:
            cls.auto_register = auto_register

        # Check for function source: either 'call' method (inheritance-based) or '_func' class attribute (decorator-based)
        # First check for decorator-based reward: _func class attribute
        if hasattr(cls, "_func") and cls._func is not None:
            # Decorator-based reward: use the function from class attribute
            self._initialize_from_function(cls._func)
        elif func is not None:
            # Legacy decorator-based reward: func passed to __init__
            self._initialize_from_function(func)
        elif hasattr(self, "call") and callable(self.call):
            # Inheritance-based reward: use the call method
            self._initialize_from_function(self.call)
        else:
            # No function source found
            self.func = None
            self.user_func = None
            self._func_sig = None

        # Determine if this reward is async
        if "_is_async_call" not in cls.__dict__:
            user_func_is_async = (
                self.user_func is not None
                and inspect.iscoroutinefunction(self.user_func)
            )
            cls._is_async_call = user_func_is_async

        # Instance-level attributes
        self.keys = None

        # Auto-register the reward instance by default (only for inheritance-based rewards)
        # Decorator-based rewards are already registered as classes in the decorator
        if cls.auto_register and cls.name and not hasattr(cls, "_func"):
            # Only register if this is an inheritance-based reward (no _func class attribute)
            # Check if already registered to avoid duplicate registration
            reward_name_lower = cls.name.lower()
            if reward_name_lower not in REWARD_REGISTRY:
                register_reward(cls.name, self)

    # ========== Function Handling ==========
    def _initialize_from_function(self, func: Callable):
        """
        Initialize the reward from a function and extract signature.

        Args:
            func: The function to use for this reward
        """
        self.func = func
        self.user_func = func

        # Get function signature for filtering kwargs
        if func is not None:
            self._func_sig = inspect.signature(func)
        else:
            self._func_sig = None

    def _filter_kwargs(self, kwargs: dict) -> dict:
        """Filter kwargs to only include parameters that the function accepts."""
        if self._func_sig is None:
            return kwargs
        return {k: v for k, v in kwargs.items() if k in self._func_sig.parameters}

    # ========== Execution ==========
    def _check_function_set(self):
        """Check if function is set, raise error if not."""
        if self.user_func is None:
            raise ValueError(
                f"Reward {self.name} has no function set. For inheritance-based rewards, define a 'call' method."
            )

    def _execute_user_function_sync(self, **kwargs):
        """Execute the user function synchronously."""
        filtered_kwargs = self._filter_kwargs(kwargs)
        return self.user_func(**filtered_kwargs)

    async def _execute_user_function_async(self, **kwargs):
        """Execute the user function, handling both sync and async functions."""
        filtered_kwargs = self._filter_kwargs(kwargs)
        if inspect.iscoroutinefunction(self.user_func):
            return await self.user_func(**filtered_kwargs)
        else:
            return self.user_func(**filtered_kwargs)

    def __call__(self, **kwargs):
        """
        Allow rewards to be called directly as functions.

        Args:
            **kwargs: All arguments as keyword arguments (e.g. context, prediction, env).

        Returns:
            reward_value_or_dict or coroutine: Coroutine if async, else direct result.
        """
        cls = type(self)
        # If async is needed, return a coroutine
        if cls._is_async_call:
            return self._call_async(**kwargs)
        else:
            # Sync call - return result directly
            return self._call_sync(**kwargs)

    def _call_sync(self, **kwargs):
        """Internal sync implementation of __call__ for non-stateful, non-async rewards."""
        self._check_function_set()

        # Execute the function
        try:
            reward_value_or_dict = self._execute_user_function_sync(**kwargs)
        except Exception as e:
            raise e  # For debugging
            reward_value_or_dict = str(e)

        # Format and return result
        return self._format_reward_result(reward_value_or_dict)

    async def _call_async(self, **kwargs):
        """Internal async implementation of __call__."""
        self._check_function_set()

        try:
            reward_value_or_dict = await self._execute_user_function_async(**kwargs)
        except Exception as e:
            raise e  # For debugging
            reward_value_or_dict = str(e)

        return self._format_reward_result(reward_value_or_dict)

    def _format_reward_result(self, reward_value_or_dict):
        """Format the reward result into the standard format."""
        if isinstance(reward_value_or_dict, dict):
            # TODO: Check if the keys are the same for all calls?
            if self.keys is None:
                self.keys = reward_value_or_dict.keys()
            return reward_value_or_dict
        elif isinstance(reward_value_or_dict, float):
            return {"reward": reward_value_or_dict}
        else:
            raise ValueError(
                f'Invalid reward: {reward_value_or_dict}, must be a float number or a dictionary with the following "reward" as a key.'
            )

    def __repr__(self):
        return f"BaseReward(name={self.name!r})"


class _CallableRewardClass(type):
    """
    Metaclass that makes Reward classes callable directly.
    When the class is called, it creates a singleton instance and calls it.
    This allows registering the class itself in REWARD_REGISTRY.
    """

    def __call__(cls, *args, **kwargs):
        # Lazy singleton pattern: create instance on first access
        # Use __dict__ to avoid triggering __getattr__
        if "_instance" not in cls.__dict__:
            cls._instance = None

        # This is a reward call - use singleton and call it
        if cls._instance is None:
            # Create instance (no func parameter allowed - func is set via call method or _func class attribute)
            cls._instance = super().__call__()

        # BaseReward.__call__ only accepts **kwargs, so raise error if positional args are provided
        if args:
            reward_name = getattr(cls, "name", cls.__name__)
            raise TypeError(
                f"Reward '{reward_name}' requires keyword arguments only. "
                f"Positional arguments are not allowed. "
                f"Use {reward_name}(arg1=value1, arg2=value2, ...) instead of {reward_name}(value1, value2, ...)."
            )

        # Call the instance - it will return either a result (sync) or coroutine (async)
        result = cls._instance(**kwargs)
        # If result is a coroutine, return it for the caller to await
        # If result is not a coroutine, return it directly
        return result

    def __getattr__(cls, name: str):
        """Delegate attribute access to singleton instance."""
        # Use __dict__ to check for _instance to avoid recursion
        if "_instance" not in cls.__dict__ or cls.__dict__.get("_instance") is None:
            # Create instance (no func parameter allowed - func is set via call method or _func class attribute)
            cls._instance = super(_CallableRewardClass, cls).__call__()
        return getattr(cls._instance, name)


def reward(
    name: Optional[str] = None,
    auto_register: bool = True,
):
    """
    Decorator that creates a BaseReward and registers it.

    Stateful rewards should accept context: Context and call
    context.acquire_resource(spec=..., scope="rollout", backend="local").
    The caller must pass context (and any other args) when invoking the reward.

    Args:
        name: The name of the reward (defaults to function name)
        auto_register: Whether to automatically register in REWARD_REGISTRY

    Returns:
        A BaseReward class (callable)
    """

    def decorator(func):
        func_name = func.__name__
        if func_name and name:
            final_name = name
        else:
            final_name = func_name

        reward_class_name = f"_Reward_{final_name}"
        reward_class = _CallableRewardClass(
            reward_class_name,
            (BaseReward,),
            {
                "name": final_name,
                "auto_register": auto_register,
                "_func": func,
            },
        )

        # Auto-register the reward class if auto_register is True
        if auto_register:
            # Register the class itself (will use singleton pattern when called)
            register_reward(final_name, reward_class)

        return reward_class

    return decorator


def register_reward(
    reward_name: str, reward_function: BaseReward | type[BaseReward]
) -> None:
    """
    Register a reward in the registry.

    Args:
        reward_name: The name of the reward
        reward_function: The reward function instance or class (for decorator-based rewards)
    """
    global REWARD_REGISTRY
    REWARD_REGISTRY[reward_name.lower()] = reward_function


def get_reward_from_name(reward_name: str) -> BaseReward | type[BaseReward]:
    """
    Get a reward function by name.

    Args:
        reward_name: Name of the reward function

    Returns:
        A BaseReward instance or class (callable in both cases)

    Raises:
        KeyError: If the reward name is not found in the registry
    """
    global REWARD_REGISTRY
    reward_name = reward_name.lower()
    if reward_name not in REWARD_REGISTRY:
        raise KeyError(
            f"Unknown reward: '{reward_name}'. Available rewards: {list(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[reward_name]


def get_rewards_from_names(
    reward_names: List[str],
) -> List[BaseReward | type[BaseReward]]:
    """
    Get multiple reward functions by name.

    Args:
        reward_names: List of reward names

    Returns:
        List of BaseReward instances or classes (callable in both cases)
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
