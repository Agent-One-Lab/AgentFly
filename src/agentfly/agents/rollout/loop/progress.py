"""The rollout progress bar, shared by the rollout strategies.

In a Ray worker (e.g. the trainer's rollout), a plain tqdm bar's redraws reach the console as
separate lines, because Ray forwards worker output line by line. There the bar uses Ray's
``tqdm_ray``: the driver draws it and updates it in place. Elsewhere it is a plain tqdm bar.
"""

import sys

from tqdm import tqdm


def in_ray_worker() -> bool:
    """True inside a Ray task or actor, not in a driver or a process without Ray."""
    ray = sys.modules.get("ray")  # a Ray worker has always imported ray already
    if ray is None:
        return False
    try:
        return ray._private.worker.global_worker.mode == ray.WORKER_MODE
    except Exception:  # private attribute: if Ray changes it, fall back to the plain bar
        return False


def progress_bar(total: int, desc: str = "Rollout"):
    """A progress bar over ``total`` items with ``update(n)`` and ``close()``."""
    if in_ray_worker():
        try:
            from ray.experimental import tqdm_ray

            return tqdm_ray.tqdm(total=total, desc=desc)
        except Exception:  # noqa: BLE001 — an experimental API; the plain bar still works
            pass
    return tqdm(total=total, desc=desc, file=sys.stdout)
