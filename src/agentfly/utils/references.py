"""Component references: resolve an agent/tool/reward by registered name OR by
import path, uniformly across the three agent-layer abstractions.

A *reference* is one of:
  - a registered name       ``"miniswe"``, ``"bash"``, ``"qa_em_reward"``
  - a module attribute      ``"pkg.module:attribute"``
  - a file attribute        ``"/path/to/file.py:attribute"`` (or ``file://…``)

Resolution: a registered name wins; otherwise the reference is imported — which
fires the target module's ``@tool`` / ``@reward`` / ``register_agent`` decorator
as a side effect — and the named attribute is returned. So resolution and
registration are the same act, and the path form composes with the existing
registries rather than replacing them.

Stdlib only (importlib): the agent layer must not depend on the training layer.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping


def is_reference(ref: str) -> bool:
    """True if ``ref`` is an import reference (has an attribute selector or looks
    like a file path) rather than a bare registered name."""
    return ":" in ref or ref.endswith(".py") or "/" in ref


def _loaded_module_for(path: Path):
    """A module already imported from ``path``, by a normal import or a dotted reference."""
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and Path(file).name == path.name and Path(file).resolve() == path:
            return module
    return None


def _import_from_file(path: str):
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"reference file not found: {p}")
    # One module per file: later references reuse it instead of executing the file again,
    # which would re-run its decorators and split its module state. The path hash keeps two
    # files with the same name (e.g. two ``tools.py``) apart.
    module_name = f"_agentfly_ref_{p.stem}_{hashlib.sha1(str(p).encode()).hexdigest()[:10]}"
    module = sys.modules.get(module_name) or _loaded_module_for(p)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from file: {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # registered before executing, as importlib does
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)  # a failed load can be retried after a fix
        raise
    # The module is importable only in this process; keep cloudpickle (and Ray's copy of it)
    # shipping its functions by value, as they were before it was registered in sys.modules.
    for name in ("cloudpickle", "ray.cloudpickle"):
        pickler = sys.modules.get(name)
        if pickler is not None and hasattr(pickler, "register_pickle_by_value"):
            pickler.register_pickle_by_value(module)
    return module


def load_object(ref: str) -> Any:
    """Import and return the object named by ``location:attribute``.

    ``location`` is a dotted module (``pkg.mod``) or a ``.py`` file path
    (absolute/relative, optional ``file://`` scheme).
    """
    location, sep, attr = ref.rpartition(":")
    if location.startswith(("file://",)):
        location = location[len("file://") :]
    if not sep or not location or not attr:
        raise ValueError(f"invalid reference {ref!r}: expected 'module_or_file.py:attribute'")
    if location.endswith(".py") or "/" in location or location.startswith("."):
        module = _import_from_file(location)
    else:
        module = importlib.import_module(location)
    if not hasattr(module, attr):
        raise AttributeError(f"reference {ref!r}: '{location}' has no attribute '{attr}'")
    return getattr(module, attr)


def resolve_reference(ref: str, registry: Mapping[str, Any], kind: str = "component") -> Any:
    """Resolve ``ref`` against ``registry`` (by lowercased name) or, on a miss,
    by importing it as a reference. Raises ``KeyError`` for an unknown bare name.
    """
    if not isinstance(ref, str):
        raise TypeError(f"{kind} reference must be a string, got {type(ref).__name__}")
    if ref.lower() in registry:
        return registry[ref.lower()]
    if is_reference(ref):
        return load_object(ref)
    raise KeyError(
        f"Unknown {kind}: {ref!r}. Available {kind}s: {sorted(registry)}. "
        f"(Use 'module:attribute' or '/path/file.py:attribute' for a custom one.)"
    )
