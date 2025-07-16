import asyncio, inspect, types, ray
from agents.tools.tool_base import Tool


# def rayify(tool: Tool, **ray_opts):
#     """
#     Wrap any `Tool` into a Ray actor that forwards the chosen coroutine
#     methods to the original instance.

#     Args:
#         tool   : The Tool object.
#         **ray_opts : Extra options passed to ray.remote().
#     Returns:
#         handle : A Ray actor handle ready to use.
#     """
#     # -------- build class dictionary dynamically --------
#     namespace: dict[str, object] = {}
#     def __init__(self):
#         self._tool = tool

#     namespace["__init__"] = __init__
#     method_names = [
#         name for name, member in inspect.getmembers(tool)
#     ]
#     additional_async_methods = ["__call__"]
#     additional_sync_methods = ["__repr__"]

#     # Get all methods of the tool
#     for name in method_names:
#         method = getattr(tool, name, None)
#         if inspect.iscoroutinefunction(method) or name in additional_async_methods:
#             async def _wrapper(self, *args, _name=name, **kwargs):
#                 return await getattr(self._tool, _name)(*args, **kwargs)
#         elif inspect.isfunction(method) or name in additional_sync_methods:
#             def _wrapper(self, *args, _name=name, **kwargs):
#                 return getattr(self._tool, _name)(*args, **kwargs)
#         else:
#             # raise ValueError(f"Method {name} is not a coroutine or function")
#             print(f"Method {name} is not a coroutine or function")
        
#         namespace[name] = _wrapper

#     # First create the class
#     ToolActorCls = types.new_class("ToolActor", (), {}, lambda ns: ns.update(namespace))
#     if ray_opts:
#         ToolActorCls = ray.remote(**ray_opts)(ToolActorCls)  # with options
#     else:
#         ToolActorCls = ray.remote(ToolActorCls)              # bare decorator form

#     return ToolActorCls.options(name=f"{tool.name}_actor").remote()

# ray_tool_wrapper.py  (v2)
import inspect, ray, types
from agents.tools.tool_base import TOOL_REGISTRY, Tool, TOOL_FACTORY

def rayify(tool: Tool, *, export=None, **ray_opts):
    """
    Turn a Tool into a Ray actor **without** pickling the Tool itself.
    We pass only its name and reconstruct from TOOL_REGISTRY inside the actor.
    """
    export = export or ("__call__", "release_env", "reset_env", "release")
    tool_name = tool.name                 # plain str ⇒ always pickleable

    # ---------- build the actor class ----------
    namespace = {}

    def __init__(self):
        # Re-lookup the tool inside the worker process
        self._tool = TOOL_FACTORY[tool_name]()

    namespace["__init__"] = __init__

    for name in export:
        meth = getattr(Tool, name, None)
        assert inspect.iscoroutinefunction(meth), f"{name} not async"

        async def _fwd(self, *a, _name=name, **kw):
            return await getattr(self._tool, _name)(*a, **kw)

        _fwd.__name__ = name
        namespace[name] = _fwd

    Actor = types.new_class("ToolActor", (), {}, lambda ns: ns.update(namespace))

    RemoteActor = ray.remote(**ray_opts)(Actor) if ray_opts else ray.remote(Actor)
    return RemoteActor.options(name=f"{tool_name}_actor").remote()

# ray_tool_runner.py  (top-level module)

# import agents.tools.utils.rayify as rayify, inspect, asyncio
# from agents.tools.tool_base import TOOL_REGISTRY, Tool
# import inspect, ray

# @ray.remote
# class ToolRunner:
#     """Ray actor that recreates a Tool by name and forwards calls to it."""
#     def __init__(self, tool_name: str):
#         # Nothing un-picklable here: we only move a string across the wire
#         self._tool = TOOL_REGISTRY[tool_name]

#     # --- generic dispatcher -------------------------------------------------
#     async def invoke(self, method: str, *args, **kw):
#         coro = getattr(self._tool, method)
#         if not inspect.iscoroutinefunction(coro):
#             raise TypeError(f"{method} is not async")
#         return await coro(*args, **kw)

#     # --- convenience wrappers for the usual calls ---------------------------
#     async def __call__(self, *args, **kw):
#         return await self.invoke("__call__", *args, **kw)

#     async def release_env(self, *a, **kw):
#         return await self.invoke("release_env", *a, **kw)

#     async def reset_env(self, *a, **kw):
#         return await self.invoke("reset_env", *a, **kw)

#     async def release(self):
#         return await self.invoke("release")

# def rayify(tool: Tool, **ray_opts):
#     """Return a Ray actor handle that runs the given tool."""
#     return ToolRunner.options(name=f"{tool.name}_runner", **ray_opts).remote(tool.name)