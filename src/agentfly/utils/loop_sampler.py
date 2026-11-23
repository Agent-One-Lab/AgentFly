"""An optional stack sampler for processes py-spy cannot attach to (Ray actors without ptrace rights).

Off unless ``AF_LOOP_SAMPLER_DIR`` is set. A daemon thread reads every thread's Python stack 20 times a second and
writes ``<dir>/<label>_<pid>.json`` every 30 s: per thread, how many samples were busy or waiting, and the busiest
functions (the innermost frame, and every frame on the stack).
"""

import collections
import json
import os
import sys
import threading
import time

INTERVAL = 0.05
WRITE_EVERY = 30.0
MAX_DEPTH = 80
TOP = 60
# The innermost function of a thread that is waiting rather than working.
WAITING = {"select", "poll", "epoll", "wait", "acquire", "sleep", "_worker", "get", "accept", "recv", "read", "main_loop",
           "run_task_loop", "_wait_for_tstate_lock", "get_objects", "get_async", "main_loop_for_threaded_actor"}
_started = False


def _name(frame) -> str:
    code = frame.f_code
    path = code.co_filename
    parts = path.split("/site-packages/")[-1].split("/src/")[-1]
    return f"{code.co_name} ({parts}:{frame.f_lineno})"


def _run(path: str, label: str) -> None:
    me = threading.get_ident()
    threads = collections.defaultdict(lambda: {"busy": 0, "waiting": 0, "self": collections.Counter(), "stack": collections.Counter()})
    started, last_write = time.time(), time.time()
    while True:
        time.sleep(INTERVAL)
        names = {t.ident: t.name for t in threading.enumerate()}
        for ident, frame in sys._current_frames().items():
            if ident == me:
                continue
            record = threads[names.get(ident, str(ident))]
            if frame.f_code.co_name in WAITING:
                record["waiting"] += 1
                continue
            record["busy"] += 1
            record["self"][_name(frame)] += 1
            seen, depth = set(), 0
            while frame is not None and depth < MAX_DEPTH:
                seen.add(_name(frame))
                frame, depth = frame.f_back, depth + 1
            record["stack"].update(seen)
        if time.time() - last_write >= WRITE_EVERY:
            last_write = time.time()
            data = {"label": label, "pid": os.getpid(), "seconds": round(last_write - started),
                    "threads": {k: {"busy": v["busy"], "waiting": v["waiting"], "self": v["self"].most_common(TOP),
                                    "stack": v["stack"].most_common(TOP)} for k, v in threads.items()}}
            with open(path + ".tmp", "w") as f:
                json.dump(data, f)
            os.replace(path + ".tmp", path)


def start_loop_sampler(label: str) -> None:
    """Start the sampler once in this process, if ``AF_LOOP_SAMPLER_DIR`` is set."""
    global _started
    directory = os.environ.get("AF_LOOP_SAMPLER_DIR")
    if not directory or _started:
        return
    _started = True
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{label}_{os.getpid()}.json")
    threading.Thread(target=_run, args=(path, label), name="af-loop-sampler", daemon=True).start()
