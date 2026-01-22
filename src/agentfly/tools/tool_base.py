import asyncio
import inspect
import json
import logging
from typing import Any, Callable, List, Optional

from ..envs.env_base import BaseEnv
from ..envs.manager.env_manager import EnvironmentManager
from .utils.schema import extract_signatures, parse_docstring, validate_schema

logger = logging.getLogger(__name__)


class BaseTool:
    """
    Universal tool wrapper that can handle both stateful and non-stateful tools.

    - For stateful tools: manages environments and pools

    - For non-stateful tools: works like a simple wrapper

    Usage patterns:

    1. Decorator-based (existing pattern):

    ```python
    @tool(name="my_tool", description="Does something")
    def my_function(arg1: str, arg2: int):
        return f"Result: {arg1} {arg2}"
    ```

    2. Inheritance-based (new pattern):

    ```python
    class MyTool(BaseTool):
        # Class-level metadata (shared across all instances)
        name = "my_tool"
        description = "A tool that uses an API key"

        def __init__(self, api_key: str):
            super().__init__()  # Class attributes are set at class definition time
            self.api_key = api_key  # Instance data
            # Schema is automatically extracted from call() method

        def call(self, query: str) -> str:
            ···
            Execute a query using the API key.

            Args:
                query (str): The query to execute.

            Returns:
                str: The result of the query.
            ···
            # Use self.api_key here
            return f"Result for {query} with key {self.api_key}"

    # Register the tool
    # Tool is automatically registered on initialization
    my_tool = MyTool(api_key="secret")
    ```
    Note: Metadata (name, description, schema, etc.) is stored as class attributes,
    making it shared across all instances of the same tool class. This is more
    memory-efficient and semantically correct since all instances of a tool type
    should have the same metadata.

    Call signature for stateful tools:

    ```python
    tool(action=..., id=...)
    ```

    Call signature for non-stateful tools:

    ```python
    tool(action=...)
    ```
    """

    # ========== Class Attributes ==========
    name: str | None = None
    description: str = ""
    schema: dict | None = None
    args: dict | None = None
    max_length: int = 2048
    status: str = "success"
    env_cls: type[BaseEnv] | None = None
    env_kwargs: dict = {}
    pool_size: int = -1  # -1 means no pool
    is_stateful: bool = False
    auto_register: bool = True

    # Class-level environment state (shared across all instances of the same tool class)
    _envs: dict[str, BaseEnv] = {}
    _locks: dict[str, asyncio.Lock] = {}
    _initialized: bool = False

    # ========== Initialization ==========
    def __init__(
        self,
    ):
        """
        Initialize a tool instance.

        Args:
            auto_register: Whether to automatically register this tool instance (defaults to True).

        Note:
            - Class attributes (name, description, schema, etc.) must be set at class definition time.
            - For inheritance-based tools, the 'call' method must be defined to provide the tool logic.
            - The 'func' parameter is not allowed - it can only be set via the 'call' method.
        """
        # Check for function source: either 'call' method (inheritance-based) or '_func' class attribute (decorator-based)
        cls = type(self)

        # First check for decorator-based tool: _func class attribute
        if hasattr(cls, "_func") and cls._func is not None:
            # Decorator-based tool: use the function from class attribute
            self._initialize_from_function(cls._func)
        elif hasattr(self, "call") and callable(self.call):
            # Inheritance-based tool: use the call method
            self._initialize_from_function(self.call)
        else:
            # No function source found
            self.func = None
            self.user_func = None
            self.is_method = False
            self.instance = None

        cls.is_stateful = cls.env_cls is not None

        # Use __dict__ to avoid triggering __getattr__ in metaclass
        if "_is_async_call" not in cls.__dict__:
            user_func_is_async = (
                self.user_func is not None
                and inspect.iscoroutinefunction(self.user_func)
            )
            cls._is_async_call = cls.is_stateful or user_func_is_async

        # Auto-register the tool instance by default
        if cls.auto_register:
            cls.register(tool_obj=self, auto_register=True)

    # ========== Function Handling ==========
    def _bind_method_tool(self, func: Callable):
        """
        Bind the method tool to the instance. We don't actually bind the instance here, we leave it to the agent to bind it.
        """
        is_method = False
        instance: Optional[Any] = None
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        if params and params[0] == "self":
            is_method = True
        return is_method, instance

    def _initialize_from_function(
        self, func: Callable, name: str | None = None, description: str | None = None
    ):
        """
        Initialize the tool from a function and extract schema.
        Sets class attributes if they don't exist, otherwise uses existing class attributes.

        Args:
            func: The function to use for this tool
            name: Optional name override (if not provided, uses func.__name__)
            description: Optional description override (if not provided, extracts from docstring)
        """
        # Check if the function is a method
        self.is_method, self.instance = self._bind_method_tool(func)

        # Set function
        self.func = func
        self.user_func = func

        cls = type(self)

        # Extract schema if not already set (check class attributes)
        if (
            "schema" not in cls.__dict__
            or cls.schema is None
            or "args" not in cls.__dict__
            or cls.args is None
        ):
            signature = extract_signatures(func)
            docs = parse_docstring(inspect.getdoc(func))
            final_name = name or getattr(cls, "name", None) or func.__name__
            final_desc = (
                description
                or docs.get("summary", "")
                or getattr(cls, "description", "")
            )

            validated_schema = validate_schema(final_name, final_desc, signature, docs)

            # Set as class attributes if not already set
            if "name" not in cls.__dict__ or cls.name is None:
                cls.name = final_name
            if "description" not in cls.__dict__ or not cls.description:
                cls.description = final_desc
            if "schema" not in cls.__dict__ or cls.schema is None:
                cls.schema = validated_schema["schema"]
            if "args" not in cls.__dict__ or cls.args is None:
                cls.args = validated_schema["args"]

    # ========== Execution ==========
    @property
    def parallel_size(self):
        if self.is_stateful:
            return self.pool_size
        # We assume/require all tools to be asyncronousable
        return 10, 000

    def _validate_call_args(self, kwargs):
        # TODO: raise error, return error message, or filter the invalid arguments, make it configurable. Currently, we just return the error message.
        for arg in kwargs:
            if arg not in self.args and not (arg == "id" and self.is_stateful):
                # raise ValueError(f"""Invalid argument "{arg}" for tool {self.name}.""")
                result = f"""Invalid argument "{arg}" for tool {self.name}."""
                return result
        return None

    def _check_function_set(self):
        """Check if function is set, raise error if not."""
        if self.user_func is None:
            raise ValueError(
                f"Tool {self.name} has no function set. For inheritance-based tools, define a 'call' method."
            )

    def _execute_user_function_sync(self, **kwargs):
        """Execute the user function synchronously. Catches user function errors and converts them to strings."""
        # Infrastructure checks (these should raise if there's a problem)
        if self.is_method:
            if self.instance is None:
                raise ValueError(f"Instance not set for method tool {self.name}")

        # Execute user function and catch errors, converting them to strings
        try:
            if self.is_method:
                return self.user_func(self.instance, **kwargs)
            else:
                return self.user_func(**kwargs)
        except Exception as e:
            # Convert user function execution errors to strings
            return str(e)

    async def _execute_user_function_async(self, **kwargs):
        """Execute the user function, handling both sync and async functions. Catches user function errors and converts them to strings."""
        # Infrastructure checks (these should raise if there's a problem)
        if self.is_method:
            if self.instance is None:
                raise ValueError(f"Instance not set for method tool {self.name}")

        # Execute user function and catch errors, converting them to strings
        try:
            if self.is_method:
                if inspect.iscoroutinefunction(self.user_func):
                    return await self.user_func(self.instance, **kwargs)
                else:
                    return self.user_func(self.instance, **kwargs)
            else:
                if inspect.iscoroutinefunction(self.user_func):
                    return await self.user_func(**kwargs)
                else:
                    return self.user_func(**kwargs)
        except Exception as e:
            # Convert user function execution errors to strings
            return str(e)

    async def _execute_stateful_tool(self, id: str, **kwargs):
        """Execute a stateful tool with environment management."""
        cls = type(self)
        await cls._initialize_envs()
        env = await cls._acquire_env(id)

        async with cls._locks[id]:
            assert kwargs.get("env", None) is None, (
                "env is not allowed to be passed to stateful tools"
            )
            kwargs["env"] = env
            return await self._execute_user_function_async(**kwargs)

    def __call__(self, **kwargs):
        """
        Call the tool with the given arguments.
        Args:
            **kwargs: The arguments to call the tool with. The arguments should be in the schema of the tool and must be specified with arg=value. For stateful tools, the id is also required for isolation.
        Returns:
            dict or coroutine: The result of the tool call. Returns a coroutine if the tool is async (stateful or has async user function), otherwise returns the result directly.
            The result is a dict with the following keys:
                - "name": The name of the tool.
                - "arguments": The arguments used to call the tool.
                - "observation": The observation of the tool call.
                - "status": The status of the tool call.
                - "info": The info of the tool call.
        """
        cls = type(self)
        # If async is needed, return a coroutine
        if cls._is_async_call:
            return self._call_async(**kwargs)
        else:
            # Sync call - return result directly
            return self._call_sync(**kwargs)

    def _call_sync(self, **kwargs):
        """Internal sync implementation of __call__ for non-stateful, non-async tools."""
        self._check_function_set()

        # Check arguments before calling the tool
        validation_error = self._validate_call_args(kwargs)
        if validation_error is not None:
            return self._format_result(validation_error, kwargs)

        # Execute the function (errors from user function are already converted to strings)
        result = self._execute_user_function_sync(**kwargs)

        # Format and return result
        return self._format_result(result, kwargs)

    async def _call_async(self, **kwargs):
        """Internal async implementation of __call__."""
        self._check_function_set()

        # Check arguments before calling the tool
        validation_error = self._validate_call_args(kwargs)
        if validation_error is not None:
            return self._format_result(validation_error, kwargs)

        # Execute the function (errors from user function are already converted to strings)
        if not self.is_stateful:
            # For non-stateful tools, directly execute the function
            result = await self._execute_user_function_async(**kwargs)
        else:
            # For stateful tools, handle environment management
            id = kwargs.pop("id", None)
            if id is None:
                result = "Error: 'id' parameter is required for stateful tools"
            else:
                result = await self._execute_stateful_tool(id, **kwargs)

        # Format and return result
        return self._format_result(result, kwargs)

    def _format_result(self, result, kwargs):
        """Format the result into the standard tool response format."""
        # Result must be a string or a dict
        if isinstance(result, str):
            if self.max_length is not None:
                result = result[: self.max_length]
            result_dict = {
                "name": self.name,
                "arguments": kwargs,
                "observation": result,
                "status": self.status,
                "info": {},
            }
            return result_dict
        elif isinstance(result, dict):
            # result should be like {"observation": "a string", "reward": 1.0}
            assert "observation" in result, (
                f"observation is required for {self.name} if tool call returns a dict"
            )
            if self.max_length is not None:
                result["observation"] = result["observation"][: self.max_length]
            observation = result.pop("observation")
            info = result
            result_dict = {
                "name": self.name,
                "arguments": kwargs,
                "observation": observation,
                "status": self.status,
                "info": info,
            }
            if "image" in result:
                result_dict["image"] = result["image"]
            return result_dict
        else:
            raise ValueError(
                f"Got invalid result: {type(result)} when calling {self.name} with arguments {kwargs}. The result should be a string or a dict containing 'observation' as a key."
            )

    # ========== Environment Management ==========
    @classmethod
    async def _initialize_envs(cls):
        """Lazy initialization of the environment pool."""
        if cls.is_stateful and not cls._initialized:
            await EnvironmentManager.start(
                cls.env_cls, size=cls.pool_size, env_kwargs=cls.env_kwargs
            )
            cls._initialized = True

    @classmethod
    def used_env_size(cls):
        """Get the number of used environments."""
        if cls.is_stateful:
            return len(cls._envs)
        return 0

    @classmethod
    def ids(cls):
        """Get the IDs of all active environments (for stateful tools only)."""
        return list(cls._envs.keys()) if cls.is_stateful else []

    @classmethod
    async def _acquire_env(cls, id: str):
        """Acquire an environment from existing environments or the pool."""
        env = cls._envs.get(id)
        if env is None:
            if not cls.is_stateful:
                return None
            env = await EnvironmentManager.acquire(cls.env_cls, id=id)
            cls._envs[id] = env
            cls._locks[id] = asyncio.Lock()
        return env

    @classmethod
    async def release(cls, id, success=True):
        """
        Release a specific environment.
        Release means we take the occupied env back, and reset it, put it back to the pool if there is one, or close it if there is no pool.
        """
        if not cls.is_stateful or id not in cls._envs:
            return

        env = cls._envs.pop(id)
        cls._locks.pop(id)
        await EnvironmentManager.release(env, id=id)

    @classmethod
    async def set_env(cls, id, env_args=None):
        """Reset a specific environment."""
        if not cls.is_stateful:
            return
        await cls._initialize_envs()
        if id in cls._envs:
            env = cls._envs[id]
            await EnvironmentManager.reset(env, env_args=env_args)
        else:
            env = await cls._acquire_env(id)
            await EnvironmentManager.reset(env, env_args=env_args)
            return

    @classmethod
    async def release_all(cls):
        """Release all environments."""
        if not cls.is_stateful:
            return

        env_ids = list(cls._envs.keys())
        await asyncio.gather(*[cls.release(env_id, success=True) for env_id in env_ids])

    # ========== Registration ==========
    @classmethod
    def register(
        cls, tool_obj=None, name: str | None = None, auto_register: bool = True
    ):
        """
        Register a tool (class or instance) in the global tool registry.

        Can be called as:
        - Class method: `Tool.register(tool_class, name="my_tool")`
        - Instance method: `tool_instance.register()` (automatically uses instance as tool_obj)

        Args:
            tool_obj: The tool class or instance to register. If None, registers cls.
                     When called on an instance, cls will be the instance's class.
            name: Optional name to register under (defaults to tool.name)
            auto_register: Whether to automatically register (defaults to True)

        Returns:
            tool_obj or cls: Returns the registered tool for method chaining
        """
        from .registry import TOOL_REGISTRY

        if not auto_register:
            return tool_obj if tool_obj is not None else cls

        # Determine which tool object to register
        if tool_obj is None:
            tool_obj = cls

        # Get the name to register under
        register_name = name or getattr(tool_obj, "name", None)
        if register_name is None:
            register_name = getattr(tool_obj, "__name__", str(tool_obj))

        # TODO: Should we warn for re-registration?
        # if register_name in TOOL_REGISTRY:
        #     warnings.warn(f"Tool {register_name!r} re-registered; overriding.")

        TOOL_REGISTRY[register_name] = tool_obj

        return tool_obj

    def __repr__(self):
        return f"<Tool name={self.name!r}, description={self.description!r}, schema={self.schema!r}>"


async def submit_tool_call(
    tool_name: str,
    tool_input: str,
    id: str = None,
    allowed_tool_names: List[str] = None,
) -> dict:
    """
    Submit a tool call to the environment.
    """
    from .registry import TOOL_REGISTRY

    if allowed_tool_names is None:
        allowed_tool_names = list(TOOL_REGISTRY.keys())

    if tool_name not in allowed_tool_names:
        tool_name = "hallucination_tool"
        tool_input = {"tool_name": str(tool_name)}

    tool_obj = TOOL_REGISTRY.get(tool_name, None)
    assert tool_obj is not None, f"Tool {tool_name} not found"
    if tool_obj.is_stateful:
        assert id is not None, "ID is required for stateful tools"
    else:
        # warnings.warn(f"ID {id} is not used for non-stateful tool {tool_name}")
        pass

    if isinstance(tool_input, str):
        """First make sure the input is a valid JSON object"""
        try:
            tool_input_json = json.loads(tool_input)
        except json.JSONDecodeError:
            tool_input_json = None
        # If the loaded input is not a dict, it means the input is not a valid JSON object
        if not isinstance(tool_input_json, dict):
            tool_input_json = None

    elif isinstance(tool_input, dict):
        tool_input_json = tool_input
    else:
        # raise ValueError(f"Invalid tool input: {tool_input}")
        # The input is not string or dict, we take it as invalid input
        tool_input_json = None

    if tool_input_json is None:
        tool_name = "invalid_input_tool"
        tool_input_json = {"tool_input": tool_input}
        tool_obj = TOOL_REGISTRY["invalid_input_tool"]

    # Add id to the input for stateful tools
    if id is not None and tool_obj.is_stateful:
        tool_input_json["id"] = id

    # Call tool_obj without await first to check if it returns a coroutine
    result = tool_obj(**tool_input_json)

    # Check if result is a coroutine and await it if needed
    if inspect.iscoroutine(result):
        # We're already in an async function, so we can directly await the coroutine
        result = await result
    # If result is not a coroutine, it's already the final value, use it directly

    return result


if __name__ == "__main__":
    # Example 1: Decorator-based tool (existing pattern)
    from .decorator import tool

    @tool(name="AdditionTool", description="Adds two numbers.")
    def add(a, b: int = 1):
        """
        Adds two numbers.

        Args:
            a (int): The first number.
            b (int): The second number which should be a non-negative integer.

        Returns:
            int: The sum of a and b.
        """
        return a + b

    @tool(description="Concatenates two strings.")
    def concat(s1, s2):
        return s1 + s2

    print("Decorator-based tool schema:")
    print(add.schema)

    # Example 2: Inheritance-based tool (new pattern)
    # Metadata can be defined as class attributes
    class APITool(BaseTool):
        """
        Example of an inheritance-based tool that stores API credentials.
        """

        # Class-level metadata (shared across all instances)
        name = "api_tool"
        description = "A tool that uses an API key to execute queries"

        def __init__(self, api_key: str):
            super().__init__()  # No need to pass metadata - uses class attributes
            self.api_key = api_key  # Instance data
            # Schema is automatically extracted from call() method

        def call(self, query: str) -> str:
            """
            Execute a query using the API key.

            Args:
                query (str): The query to execute.

            Returns:
                str: The result of the query.
            """
            # Use self.api_key here
            return f"Result for '{query}' using API key: {self.api_key[:5]}..."

    # Tool is automatically registered on initialization
    api_tool = APITool(api_key="secret_key_12345")

    print("Concat tool: ", concat)
    print("Type of concat: ", type(concat))

    result = concat(s1="Hello", s2="World")
    print("Result: ", result)

    print("API tool: ", api_tool)
    print("Type of api_tool: ", type(api_tool))
    result = api_tool(query="What is the weather in Tokyo?")
    print("Result: ", result)
