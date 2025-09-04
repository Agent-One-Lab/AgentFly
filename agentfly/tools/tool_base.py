import asyncio
import inspect
import json
import time
import warnings
from ..envs.manager.warm_pool import WarmPool
from ..envs.env_base import BaseEnv
from .utils.schema import extract_signatures, parse_docstring, validate_schema
from .utils.runner import syncronize
from typing import Callable, Dict, List
import contextvars
from . import TOOL_REGISTRY, TOOL_FACTORY
import concurrent.futures
from ..envs.manager.env_manager import EnvironmentManager

import logging

logger = logging.getLogger(__name__)

# current_env = contextvars.ContextVar("current_env")

class Tool:
    """
    Universal tool wrapper that can handle both stateful and non-stateful tools.

    - For stateful tools: manages environments and pools
    
    - For non-stateful tools: works like a simple wrapper
    
    
    Call signature for stateful tools:
    
    .. code-block:: python

        tool(action=..., id=...)

    Call signature for non-stateful tools:
    
    .. code-block:: python

        tool(action=...)
    """
    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        schema: dict | None = None,
        args: dict | None = None,
        max_length: int = 2048,
        env_cls: type[BaseEnv] | None = None,
        env_kwargs: dict | None = None,
        pool_size: int = -1, # -1, or 0 means no pool
        stateful: bool = False,
        status: str = "success"
    ):
        # Basic properties
        self.func = func
        self.name = name or func.__name__
        self.description = description or ""
        self.schema = schema
        self.args = args
        self.max_length = max_length
        self.status = status
        
        # Stateful properties
        self.env_cls, self.env_kwargs = env_cls, env_kwargs or {}
        self.pool_size = pool_size
        # self._pool = None
        # self._pool_initialized = False
        self.initialized = False
        self.is_stateful = stateful
        self._envs: dict[str, BaseEnv] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.user_func = func
        
    @property
    def parallel_size(self):
        if self.is_stateful:
            return self.pool_size
        # We assume/require all tools to be asyncronousable
        return 10,000
    
    async def _initialize_envs(self):
        # Lazy initialization of the pool
        if self.is_stateful and not self.initialized:
            await EnvironmentManager.start(self.env_cls, size=self.pool_size, env_kwargs=self.env_kwargs)
            self.initialized = True

    @property
    def used_env_size(self):
        if self.is_stateful:
            return len(self._envs)
        return 0

    def _validate_call_args(self, kwargs):
        # TODO: raise error, return error message, or filter the invalid arguments, make it configurable. Currently, we just return the error message.
        for arg in kwargs:
            if arg not in self.args and not (arg == "id" and self.is_stateful):
                # raise ValueError(f"""Invalid argument "{arg}" for tool {self.name}.""")
                result = f"""Invalid argument "{arg}" for tool {self.name}."""
                return result
        return None
        
    async def __call__(self, **kwargs):
        """
        Call the tool with the given arguments.
        Args:
            **kwargs: The arguments to call the tool with. The arguments should be in the schema of the tool and must be specified with arg=value. For stateful tools, the id is also required for isolation.
        Returns:
            dict: The result of the tool call. The result is a dict with the following keys:
                - "name": The name of the tool.
                - "arguments": The arguments used to call the tool.
                - "observation": The observation of the tool call.
                - "status": The status of the tool call.
                - "info": The info of the tool call.
        """
        # Check arguments before calling the tool
        result = self._validate_call_args(kwargs)

        # If the arguments are valid, call the tool
        if result is None:
            try:
                if not self.is_stateful:
                    # For non-stateful tools, directly execute the function
                    result = await self.user_func(**kwargs) if inspect.iscoroutinefunction(self.user_func) \
                            else self.user_func(**kwargs)
                else:
                    # For stateful tools, handle environment management
                    id = kwargs.pop('id', None)
                    if id is None:
                        result = "Error: 'id' parameter is required for stateful tools"
                    else:
                        await self._initialize_envs()
                        env = await self._acquire_env(id)
                        
                        async with self._locks[id]:
                            # token = current_env.set(env)
                            assert kwargs.get("env", None) is None, "env is not allowed to be passed to stateful tools"
                            try:
                                result = await self.user_func(env=env,**kwargs) if inspect.iscoroutinefunction(self.user_func) \
                                    else self.user_func(**kwargs)
                            finally:
                                pass
            except Exception as e:
                result = str(e)
        # If the arguments are invalid, simply use the result from the validation
        else:
            pass

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
            assert "observation" in result, f"observation is required for {self.name} if tool call returns a dict"
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
            return result_dict
        else:
            raise ValueError(f"Got invalid result: {type(result)} when calling {self.name} with arguments {kwargs}. The result should be a string or a dict containing 'observation' as a key.")
        

    def call(self, **kwargs):
        """Synchronous wrapper for the async __call__ method."""
        """
        TODO: We need to add more backend asyncronous support
        - Currently, we only support threading
        - We should also support multiprocessing
        """
        return syncronize(self.__call__(**kwargs))

    @property
    def ids(self):
        """Get the IDs of all active environments (for stateful tools only)."""
        return list(self._envs.keys()) if self.is_stateful else []

    async def _acquire_env(self, id: str):
        """Acquire an environment from existing environments or the pool."""
        env = self._envs.get(id)
        if env is None:
            if not self.is_stateful:
                return None
            env = await EnvironmentManager.acquire(self.env_cls, id=id)
            self._envs[id] = env
            self._locks[id] = asyncio.Lock()
        return env
        
    # Release means we take the occupied env back, and reset it, put it back to the pool if there is one, or close it if there is no pool
    async def release(self, id, success=True):
        """Release a specific environment."""
        if not self.is_stateful or id not in self._envs:
            return
            
        env = self._envs.pop(id)
        self._locks.pop(id)
        await EnvironmentManager.release(env, id=id)

    async def set_env(self, id, env_args=None):
        """Reset a specific environment."""
        if not self.is_stateful:
            return
        await self._initialize_envs()
        if id in self._envs:
            env = self._envs[id]
            await EnvironmentManager.reset(env, env_args=env_args)
        else:
            env = await self._acquire_env(id)
            await EnvironmentManager.reset(env, env_args=env_args)
            return

    async def release_all(self):
        """Release all environments."""
        if not self.is_stateful:
            return
            
        env_ids = list(self._envs.keys())
        await asyncio.gather(*[self.release_env(env_id, success=True) for env_id in env_ids])

                
    def __repr__(self):
        return f"<Tool name={self.name!r}, description={self.description!r}, schema={self.schema!r}>"


def tool(
    name: str | None = None,
    description: str | None = None,
    status: str = "success",
    max_length: int = 2048,
    auto_register: bool = True,
    stateful: bool = False,
    env_cls: type[BaseEnv] | None = None,
    env_kwargs: dict | None = None,
    pool_size: int = -1, # -1, or 0 means no pool
):
    """
    Decorator that registers a callable as a tool.
    Creates a Tool instance that can handle both stateful and non-stateful behavior.

    Args:
        name (str): The name of the tool.
        description (str): The description of the tool.
        status (str): We use this to control the chain search workflow.
            - "terminal": The tool call is the final step in the chain. The search will be stopped.
            - "continue": The tool call is not the final step in the chain. The search will continue.
        max_length (int): The maximum length of the tool's output/observation.
        auto_register (bool): Whether to automatically register the tool. This is used to get tool by name.
        stateful (bool): Whether the tool is stateful. A stateful tool is a tool that manages its own environment.
        env_cls (type[BaseEnv]): The environment class for the tool.
        env_kwargs (dict): The kwargs for the environment class.
        pool_size (int): The size of the pool for the environment.
    """
    def decorator(func):
        nonlocal name, description

        # ── name and description
        func_name = func.__name__
        final_name = name or func_name
        if name and name != func_name:
            logger.warning(f"Tool name {func_name!r} overridden by {name!r}")
            # warnings.warn(f"Tool name {func_name!r} overridden by {name!r}")

        signature  = extract_signatures(func)
        docs       = parse_docstring(inspect.getdoc(func))
        final_desc = description or docs.get("summary", "")
        validated_schema = validate_schema(final_name, final_desc, signature, docs)

        # Create the tool
        def factory():
            return Tool(
                func=func,
                name=final_name,
                description=final_desc,
                schema=validated_schema["schema"],
                args=validated_schema["args"],
                max_length=max_length,
                env_cls=env_cls,
                env_kwargs=env_kwargs,
                pool_size=pool_size,
                stateful=stateful or env_cls is not None,
                status=status,
            )
        tool_obj = factory()

        # auto-registration
        if auto_register:
            if final_name in TOOL_REGISTRY:
                warnings.warn(f"Tool {final_name!r} re-registered; overriding.")
            TOOL_REGISTRY[final_name] = tool_obj
            TOOL_FACTORY[final_name] = factory

        return tool_obj

    return decorator


async def submit_tool_call(
    tool_name: str,
    tool_input: str,
    id: str=None,
    allowed_tool_names: List[str] = None,
) -> dict:
    """
    Submit a tool call to the environment.
    """
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
    
    return await tool_obj(**tool_input_json)


def submit_tool_calls(
    tool_names: List[str],
    tool_inputs: List[Dict | str],
    ids: List[str],
    allowed_tool_names: List[str] = None,
) -> List[dict]:
    """
    Submit tool calls to the environment. This is a synchronous wrapper that blocks until all results are ready.
    Uses ThreadPoolExecutor to run tool calls in parallel.
    """

    if allowed_tool_names is None:
        allowed_tool_names = list(TOOL_REGISTRY.keys())


    mapped_tool_names = []
    mapped_tool_inputs = []
    tool_objs = []

    for tool_name, tool_input, id in zip(tool_names, tool_inputs, ids):
        if isinstance(tool_input, dict):
            tool_input_json = tool_input
        elif isinstance(tool_input, str):
            try:
                tool_input_json = json.loads(tool_input)
            except json.JSONDecodeError:
                tool_input_json = None
        else:
            raise ValueError(f"Invalid tool input: {tool_input}")

        
        if tool_name not in allowed_tool_names:
            # Called a non-existent tool
            mapped_tool_name = "hallucination_tool"
            tool_input_json = {"tool_name": tool_name}
        elif tool_input_json is None:
            # Invalid input
            mapped_tool_name = "invalid_input_tool"
            tool_input_json = {"tool_input": tool_input}
        else:
            mapped_tool_name = tool_name
        
        tool_obj = TOOL_REGISTRY[mapped_tool_name]
        if tool_obj.is_stateful:
            assert id is not None, "ID is required for stateful tools"
            tool_input_json["id"] = id
        else:
            if id is not None:
                warnings.warn(f"ID {id} is not used for non-stateful tool {mapped_tool_name}")

        mapped_tool_names.append(mapped_tool_name)
        mapped_tool_inputs.append(tool_input_json)
        tool_objs.append(tool_obj)

    # Use ThreadPoolExecutor to run tool calls in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a list of futures
        futures = [
            executor.submit(tool_obj.call, **tool_input)
            for tool_obj, tool_input in zip(tool_objs, mapped_tool_inputs)
        ]
        
        # Wait for all futures to complete and get results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return results


@tool()
def hallucination_tool(tool_name):
    return f"Hallucinated tool: {tool_name} does not exist."

@tool()
def invalid_input_tool(tool_input):
    return f"Invalid input: {tool_input}, input must be a valid JSON object."



if __name__ == "__main__":
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
    print(add.schema)
