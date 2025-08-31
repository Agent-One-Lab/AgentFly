import asyncio, inspect, types, ray
from ...tools.tool_base import TOOL_FACTORY, Tool

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
