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

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Mapping


def is_reference(ref: str) -> bool:
    """True if ``ref`` is an import reference (has an attribute selector or looks
    like a file path) rather than a bare registered name."""
    return ":" in ref or ref.endswith(".py") or "/" in ref


def _import_from_file(path: str):
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"reference file not found: {p}")
    module_name = f"_agentfly_ref_{p.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from file: {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_object(ref: str) -> Any:
    """Import and return the object named by ``location:attribute``.

    ``location`` is a dotted module (``pkg.mod``) or a ``.py`` file path
    (absolute/relative, optional ``file://`` scheme).
    """
    location, sep, attr = ref.rpartition(":")
    if location.startswith(("file://",)):
        location = location[len("file://"):]
    if not sep or not location or not attr:
        raise ValueError(
            f"invalid reference {ref!r}: expected 'module_or_file.py:attribute'"
        )
    if location.endswith(".py") or "/" in location or location.startswith("."):
        module = _import_from_file(location)
    else:
        module = importlib.import_module(location)
    if not hasattr(module, attr):
        raise AttributeError(
            f"reference {ref!r}: '{location}' has no attribute '{attr}'"
        )
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
