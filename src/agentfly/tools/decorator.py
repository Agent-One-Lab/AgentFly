"""The ``@tool`` decorator: turn a function into a registered tool.

``tool(name=..., description=...)`` builds a :class:`~agentfly.tools.tool_base.BaseTool` subclass
around the function, derives the JSON schema from its signature and docstring, and registers it
in the tool registry under ``name``. The resulting class is callable like the function
(``calculator(expression="1+1")``) through a lazily created singleton instance.
"""

import inspect
import logging

from .utils.schema import extract_signatures, parse_docstring, validate_schema

logger = logging.getLogger(__name__)


class _CallableToolClass(type):
    """
    Metaclass that makes Tool classes callable directly.
    When the class is called, it creates a singleton instance and calls it.
    This allows registering the class itself in TOOL_REGISTRY.
    """

    def __call__(cls, *args, **kwargs):
        # Lazy singleton pattern: create instance on first access
        # Use __dict__ to avoid triggering __getattr__
        if "_instance" not in cls.__dict__:
            cls._instance = None

        # This is a tool call - use singleton and call it
        if cls._instance is None:
            # Create instance (no func parameter allowed - func is set via call method or _func class attribute)
            cls._instance = super().__call__()

        # Tool.__call__ only accepts **kwargs, so raise error if positional args are provided
        if args:
            tool_name = getattr(cls, "name", cls.__name__)
            raise TypeError(
                f"Tool '{tool_name}' requires keyword arguments only. "
                f"Positional arguments are not allowed. "
                f"Use {tool_name}(arg1=value1, arg2=value2, ...) instead of {tool_name}(value1, value2, ...)."
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
            cls._instance = super(_CallableToolClass, cls).__call__()
        return getattr(cls._instance, name)


def tool(
    name: str | None = None,
    description: str | None = None,
    status: str = "success",
    max_length: int = 2048,
    auto_register: bool = True,
):
    """
    Decorator that registers a callable as a tool.

    A tool that keeps state across calls (a game, a sandbox) declares a ``context: Context``
    parameter and acquires its environment with ``context.acquire_resource(...)``; see
    docs/start/custom_environment.md.

    Args:
        name (str): The name of the tool.
        description (str): The description of the tool.
        status (str): We use this to control the chain search workflow.
            - "terminal": The tool call is the final step in the chain. The search will be stopped.
            - "continue": The tool call is not the final step in the chain. The search will continue.
        max_length (int): The maximum length of the tool's output/observation.
        auto_register (bool): Whether to automatically register the tool. This is used to get tool by name.
    """
    from .tool_base import BaseTool

    def decorator(func):
        nonlocal name, description

        # ── name and description
        func_name = func.__name__
        final_name = name or func_name
        if name and name != func_name:
            logger.debug(f"Tool name {func_name!r} overridden by {name!r}")

        signature = extract_signatures(func)
        docs = parse_docstring(inspect.getdoc(func))
        final_desc = description or docs.get("description", "")
        validated_schema = validate_schema(final_name, final_desc, signature, docs)

        # Create a Tool subclass with class-level metadata attributes
        # This ensures all instances share the same metadata
        # Use _CallableToolClass metaclass to make the class itself callable
        # The class inherits from Tool, so isinstance/issubclass checks work correctly
        # This makes decorator-based tools compatible with inheritance-based tools
        tool_class_name = f"_Tool_{final_name}"
        tool_class = _CallableToolClass(
            tool_class_name,
            (BaseTool,),  # Inherit from Tool class - ensures isinstance(tool_class(), Tool) == True
            {
                "name": final_name,
                "description": final_desc,
                "schema": validated_schema["schema"],
                "args": validated_schema["args"],
                "max_length": max_length,
                "status": status,
                # Store the function as a class attribute for reference
                "_func": func,
            },
        )

        # auto-registration using Tool.register classmethod
        if auto_register:
            BaseTool.register(tool_obj=tool_class, name=final_name, auto_register=True)

        return tool_class

    return decorator
