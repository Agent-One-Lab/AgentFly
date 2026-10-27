#!/usr/bin/env python
"""snippet_runner.py  –  ONE process per user snippet.

• Reads code from STDIN (no files needed).
• Sets hard RLIMIT_AS + RLIMIT_CPU.
• Provides a permissive   __import__   that blocks only
  obviously dangerous modules (ctypes, multiprocessing, …).
• On success:  prints user stdout to STDOUT, exits 0.
• On error:    prints message to STDERR, exits non-zero.
"""

import ast
import builtins
import os
import resource
import sys

# --- 1 ⚙️  configuration (in sync with parent / cgroup) ----------
MAX_RSS = int(os.environ.get("CHILD_MEM", 1 * 2**30))  # 1 GiB
MAX_CPUT = int(os.environ.get("MAX_CPU_SECS", 8))  # CPU seconds
BAD_ROOTS = {  # minimal blacklist
    "ctypes",
    "cffi",
    "multiprocessing",
    "resource",
    "sys",
    "signal",
    "importlib.machinery",
    "subprocess",
    "socketserver",
    "ssl",
    "pty",
    "fcntl",
}

# --- 2 🔒  prepare the restricted built-ins ----------------------
_real_import = builtins.__import__  # keep original for proxy


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.partition(".")[0]
    if root in BAD_ROOTS:
        raise ImportError(f"Module '{root}' is blocked in this sandbox")
    return _real_import(name, globals, locals, fromlist, level)


def _safe_builtins():
    safe = dict(builtins.__dict__)  # make a *copy*
    safe["__import__"] = _restricted_import
    return safe  # <-- return dict, not mappingproxy


# Numeric libraries (numpy's OpenBLAS, MKL, OpenMP) start one thread per CPU core by default;
# on a many-core machine those threads alone exceed the memory limit and the process is killed.
# A snippet gets a few CPU-seconds anyway, so one thread is enough.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# --- 3 🚦  runtime hard-limits (address space + CPU) -------------
resource.setrlimit(resource.RLIMIT_AS, (MAX_RSS, MAX_RSS))
resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPUT, MAX_CPUT))

# --- 4 📥  read the user code from STDIN -------------------------
src = sys.stdin.read()

try:
    parsed = ast.parse(src, "<user>", "exec")
    #  (optional) quick AST sanity checks could go here
    # One namespace for globals and locals, as in a normal script: functions can then see
    # top-level names (other functions, imports), and `if __name__ == "__main__":` runs.
    exec_env = {"__builtins__": _safe_builtins(), "__name__": "__main__"}
    exec(compile(parsed, "<user>", "exec"), exec_env)
except MemoryError:
    print("Memory limit exceeded", file=sys.stderr)
    sys.exit(1)
except BaseException as e:
    # propagate any Python error to parent via STDERR
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)

sys.stdout.flush()
sys.stderr.flush()
sys.exit(0)
