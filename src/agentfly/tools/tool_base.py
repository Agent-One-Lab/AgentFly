import asyncio
import inspect
import json
import logging
from typing import Any, Callable, List, Optional
from .utils.schema import extract_signatures, parse_docstring, validate_schema
from .. import TOOL_ERROR_AS_OBSERVATION

logger = logging.getLogger(__name__)


class BaseTool:
    """
    Universal tool wrapper for both sync and async tools.

    Tools can access rollout context (rollout_id, group_id, metadata) and resources
    via the Context parameter, which is automatically injected when requested.

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

    3. Tools with Context (for resource access):

    ```python
    @tool(name="grep_search")
    async def grep_search(pattern: str, path: str = ".", context: "Context"):
        container = await context.acquire_resource(image_id=..., scope="rollout")
        return container.run_cmd(f"grep -r {pattern} {path}")
    ```
    """

    # ========== Class Attributes ==========
    name: str | None = None
    description: str = ""
    schema: dict | None = None
    args: dict | None = None
    max_length: int = 2048
    status: str = "success"
    auto_register: bool = True

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

        # Use __dict__ to avoid triggering __getattr__ in metaclass
        if "_is_async_call" not in cls.__dict__:
            user_func_is_async = (
                self.user_func is not None
                and inspect.iscoroutinefunction(self.user_func)
            )
            cls._is_async_call = user_func_is_async

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
        # We assume/require all tools to be asyncronousable
        return 10_000

    def _validate_call_args(self, kwargs):
        # TODO: raise error, return error message, or filter the invalid arguments, make it configurable. Currently, we just return the error message.
        # Context is injected by the rollout and is not in schema
        injected_params = ("context",)
        for arg in kwargs:
            if arg not in self.args and arg not in injected_params:
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
        """Execute the user function synchronously.

        On error: returns str(e) if ``TOOL_ERROR_AS_OBSERVATION`` is True,
        otherwise re-raises.
        """
        # Infrastructure checks (these should raise if there's a problem)
        if self.is_method:
            if self.instance is None:
                raise ValueError(f"Instance not set for method tool {self.name}")

        try:
            if self.is_method:
                return self.user_func(self.instance, **kwargs)
            else:
                return self.user_func(**kwargs)
        except Exception as e:
            if TOOL_ERROR_AS_OBSERVATION:
                return str(e)
            raise

    async def _execute_user_function_async(self, **kwargs):
        """Execute the user function, handling both sync and async functions.

        On error: returns str(e) if ``agentfly.TOOL_ERROR_AS_OBSERVATION`` is True,
        otherwise re-raises.
        """
        # Infrastructure checks (these should raise if there's a problem)
        if self.is_method:
            if self.instance is None:
                raise ValueError(f"Instance not set for method tool {self.name}")

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
            if TOOL_ERROR_AS_OBSERVATION:
                return str(e)
            raise

    def __call__(self, **kwargs):
        """
        Call the tool with the given arguments.
        Args:
            **kwargs: The arguments to call the tool with. The arguments should be in the schema of the tool and must be specified with arg=value.
        Returns:
            dict or coroutine: The result of the tool call. Returns a coroutine if the tool is async, otherwise returns the result directly.
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
        """Internal sync implementation of __call__ for sync tools."""
        self._check_function_set()

        # Check arguments before calling the tool
        validation_error = self._validate_call_args(kwargs)
        if validation_error is not None:
            return self._format_result(validation_error, kwargs)

        # Execute the function (may convert errors to strings; see TOOL_ERROR_AS_OBSERVATION)
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

        # Execute the function (may convert errors to strings; see TOOL_ERROR_AS_OBSERVATION)
        result = await self._execute_user_function_async(**kwargs)

        # Format and return result
        return self._format_result(result, kwargs)

    def _format_result(self, result, kwargs):
        """Format the result into the standard tool response dict.

        Normalization is delegated to :meth:`ToolResult.from_raw`; the legacy
        dict shape is reproduced via :meth:`ToolResult.to_dict` so existing
        downstream consumers keep working.
        """
        from .types import ToolResult

        return ToolResult.from_raw(
            result,
            name=self.name,
            arguments=kwargs,
            status=self.status,
            max_length=self.max_length,
        ).to_dict()

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


def _tool_accepts_context(tool_obj: Any) -> bool:
    """Check if the tool's function accepts a context parameter."""
    func = None
    if hasattr(tool_obj, "user_func") and tool_obj.user_func is not None:
        func = tool_obj.user_func
    elif hasattr(tool_obj, "_func") and tool_obj._func is not None:
        func = tool_obj._func  # Decorator-based: func stored on class
    if func is not None:
        sig = inspect.signature(func)
        return "context" in sig.parameters
    return False


async def submit_tool_call(
    tool_name: str,
    tool_input: str,
    context: Optional[Any] = None,
    allowed_tool_names: Optional[List[str]] = None,
) -> dict:
    """
    Submit a tool call to the environment.

    Args:
        tool_name: Name of the tool to call.
        tool_input: JSON string or dict of tool arguments.
        context: Optional Context instance for rollout-scoped data and resources.
                 Injected into tools that accept a `context` parameter.
        allowed_tool_names: Optional list of allowed tool names.

    Returns:
        dict: Tool result with keys like observation, status, etc.
    """
    from .registry import TOOL_REGISTRY

    if allowed_tool_names is None:
        allowed_tool_names = list(TOOL_REGISTRY.keys())

    if tool_name not in allowed_tool_names:
        tool_name = "hallucination_tool"
        tool_input = {"tool_name": str(tool_name)}

    tool_obj = TOOL_REGISTRY.get(tool_name, None)
    assert tool_obj is not None, f"Tool {tool_name} not found"

    if isinstance(tool_input, str):
        try:
            tool_input_json = json.loads(tool_input)
        except json.JSONDecodeError:
            tool_input_json = None
        if not isinstance(tool_input_json, dict):
            tool_input_json = None

    elif isinstance(tool_input, dict):
        tool_input_json = tool_input
    else:
        tool_input_json = None

    if tool_input_json is None:
        tool_name = "invalid_input_tool"
        tool_input_json = {"tool_input": tool_input}
        tool_obj = TOOL_REGISTRY["invalid_input_tool"]

    # Inject Context if the tool accepts it
    if context is not None and _tool_accepts_context(tool_obj):
        tool_input_json["context"] = context

    # Call tool_obj without await first to check if it returns a coroutine
    result = tool_obj(**tool_input_json)

    # Check if result is a coroutine and await it if needed
    if inspect.iscoroutine(result):
        result = await result

    return result


if __name__ == "__main__":
    # Example 1: Decorator-based tool (existing pattern)
    from .decorator import tool

    # --8<-- [start:addition_tool_example]
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
    # --8<-- [end:addition_tool_example]

    @tool(description="Concatenates two strings.")
    def concat(s1, s2):
        return s1 + s2

    print("Decorator-based tool schema:")
    print(add.schema)

    # Example 2: Inheritance-based tool (new pattern)
    # Metadata can be defined as class attributes
    # --8<-- [start:api_tool_example]
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
    # --8<-- [end:api_tool_example]

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
