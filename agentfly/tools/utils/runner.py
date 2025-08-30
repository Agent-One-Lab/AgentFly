import atexit
import asyncio, threading
from typing import Coroutine, Any

# ---------- one global background loop ----------
_loop_holder = {"loop": None, "thread": None}

# atexit.register(_loop_holder["loop"].stop)

def _ensure_background_loop():
    if _loop_holder["loop"] is None:
        loop = asyncio.new_event_loop()
        t = threading.Thread(
            target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()),
            name="tool-bg-loop",
            daemon=True,
        )
        t.start()
        _loop_holder["loop"], _loop_holder["thread"] = loop, t
    return _loop_holder["loop"]

def syncronize(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Run *coro* in the single background event-loop and block
    until it completes, regardless of caller context.
    """
    # Fast-path: caller already in *our* background loop → just await
    loop = _ensure_background_loop()
    try:
        running = asyncio.get_running_loop()
        if running is loop:
            # We are *inside* the bg loop
            return coro  # caller must await
    except RuntimeError:
        pass  # no running loop

    # Submit to the bg loop and wait
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()
