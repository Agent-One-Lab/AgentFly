"""
File manager with tool-based editing for SWE-bench style tasks.

Designed for parallel agentic RL training. Each container subprocess builds a
fresh :class:`FileManager` (state such as undo lives on disk under the workspace).

Search: system grep with Python fallback (no ripgrep — grep is faster
in Docker containers with limited cores and concurrent tasks).

Undo: per-file snapshot stack under ``.agentfly_undo/`` inside the workspace
so each container invocation (copied ``file_manager.py``) can still undo edits.
"""

import hashlib
import os
import sys
import json
import re
import fnmatch
import shutil
import subprocess
from typing import Optional, List, Tuple


# ═════════════════════════════════════════════════════════════════════
#  FileManager — one instance per task
# ═════════════════════════════════════════════════════════════════════

class FileManager:

    # Undo snapshots live under workspace root (hidden; skipped by list/grep walks).
    _UNDO_DIRNAME = ".agentfly_undo"

    # Directories to skip during traversal
    _SKIP_DIRS = frozenset({
        "__pycache__", "node_modules", ".tox", ".eggs",
        "build", "dist", ".mypy_cache", ".pytest_cache",
    })

    # Binary file extensions to skip
    _BINARY_EXTS = frozenset({
        ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll", ".exe",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
        ".woff", ".woff2", ".ttf", ".eot",
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".pickle", ".pkl", ".npy", ".npz", ".h5", ".hdf5",
        ".db", ".sqlite", ".sqlite3", ".mo",
    })

    def __init__(self, task_id: str, workspace_root: Optional[str] = None):
        self.task_id = task_id
        root = workspace_root or os.environ.get("WORKSPACE_ROOT", "/testbed")
        self.root = os.path.realpath(os.path.abspath(root))
        self._file_index_cache: Optional[List[str]] = None

    # ── Internal helpers ─────────────────────────────────────────────

    def _undo_dir_for_file(self, full: str) -> str:
        """Directory holding numbered snapshot files for one workspace file."""
        rel = os.path.relpath(full, self.root)
        rel_key = rel.replace(os.sep, "/")
        h = hashlib.sha256(rel_key.encode("utf-8")).hexdigest()[:32]
        return os.path.join(self.root, self._UNDO_DIRNAME, h)

    def _list_undo_snapshot_names(self, undo_dir: str) -> List[int]:
        if not os.path.isdir(undo_dir):
            return []
        out: List[int] = []
        for name in os.listdir(undo_dir):
            if name.isdigit():
                out.append(int(name))
        return sorted(out)

    def _push_undo_snapshot(self, full: str, content: str) -> None:
        undo_dir = self._undo_dir_for_file(full)
        os.makedirs(undo_dir, exist_ok=True)
        rel = os.path.relpath(full, self.root)
        with open(os.path.join(undo_dir, "_source_relpath.txt"), "w", encoding="utf-8") as meta:
            meta.write(rel.replace(os.sep, "/"))
        nums = self._list_undo_snapshot_names(undo_dir)
        n = (max(nums) + 1) if nums else 1
        snap = os.path.join(undo_dir, f"{n:08d}")
        with open(snap, "w", encoding="utf-8") as f:
            f.write(content)

    def _pop_undo_snapshot(self, full: str) -> Optional[str]:
        undo_dir = self._undo_dir_for_file(full)
        nums = self._list_undo_snapshot_names(undo_dir)
        if not nums:
            return None
        top = max(nums)
        snap = os.path.join(undo_dir, f"{top:08d}")
        with open(snap, "r", encoding="utf-8") as f:
            data = f.read()
        os.remove(snap)
        try:
            if not self._list_undo_snapshot_names(undo_dir):
                for name in ("_source_relpath.txt",):
                    p = os.path.join(undo_dir, name)
                    if os.path.isfile(p):
                        os.remove(p)
                os.rmdir(undo_dir)
        except OSError:
            pass
        return data

    def _clear_undo_for_file(self, full: str) -> None:
        undo_dir = self._undo_dir_for_file(full)
        if os.path.isdir(undo_dir):
            shutil.rmtree(undo_dir, ignore_errors=True)

    def _undo_stats(self) -> Tuple[int, int]:
        """(files_with_snapshots, total_snapshot_files)."""
        base = os.path.join(self.root, self._UNDO_DIRNAME)
        if not os.path.isdir(base):
            return 0, 0
        files_with = 0
        total = 0
        for name in os.listdir(base):
            sub = os.path.join(base, name)
            if not os.path.isdir(sub):
                continue
            n = len(self._list_undo_snapshot_names(sub))
            if n:
                files_with += 1
                total += n
        return files_with, total

    def _safe_path(self, path: str) -> str:
        """Resolve path relative to workspace root; block escapes."""
        if path.startswith("/"):
            path = path[1:] or "."
        full = os.path.realpath(os.path.abspath(os.path.join(self.root, path)))
        if not full.startswith(self.root):
            raise PermissionError("Access denied: path is outside workspace.")
        return full

    @classmethod
    def _skip_dir(cls, name: str) -> bool:
        return name.startswith(".") or name in cls._SKIP_DIRS or name.endswith(".egg-info")

    @classmethod
    def _is_binary(cls, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in cls._BINARY_EXTS

    def _get_file_index(self) -> List[str]:
        """Cached list of all text files in the workspace."""
        if self._file_index_cache is not None:
            return self._file_index_cache
        index = []
        for root_dir, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not self._skip_dir(d)]
            for f in sorted(files):
                fp = os.path.join(root_dir, f)
                if not self._is_binary(fp):
                    index.append(fp)
        self._file_index_cache = index
        return index

    # ── Tools: read ──────────────────────────────────────────────────

    def read_file(self, path: str, start_line: Optional[int] = None,
                  end_line: Optional[int] = None) -> str:
        """Read a file with numbered lines. Optionally show a line range."""
        full = self._safe_path(path)
        with open(full, "r") as f:
            lines = f.readlines()

        total = len(lines)
        s = max(1, start_line or 1)
        e = min(total, end_line or total)

        numbered = [f"{i+1:6d} | {lines[i]}" for i in range(s - 1, e)]
        rel = os.path.relpath(full, self.root)
        header = f"[File: {rel} ({total} lines)]"
        if start_line or end_line:
            header += f" [Showing {s}-{e}]"
        return header + "\n" + "".join(numbered)

    # ── Tools: navigate ──────────────────────────────────────────────

    def list_files(self, path: str = ".", max_depth: int = 3) -> str:
        """List files in directory, skipping hidden/junk dirs and binaries."""
        full = self._safe_path(path)
        base_depth = full.rstrip(os.sep).count(os.sep)
        result = []
        for root_dir, dirs, files in os.walk(full):
            dirs[:] = sorted(d for d in dirs if not self._skip_dir(d))
            if root_dir.rstrip(os.sep).count(os.sep) - base_depth >= max_depth:
                dirs.clear()
                continue
            for f in sorted(files):
                fp = os.path.join(root_dir, f)
                if not self._is_binary(fp):
                    result.append(os.path.relpath(fp, self.root))
        return "\n".join(result) if result else "(empty)"

    def grep_search(self, pattern: str, path: str = ".",
                    include: Optional[str] = None,
                    max_results: int = 1000) -> str:
        """
        Search for a regex pattern across files.
        Uses system grep when available, falls back to Python.

        Args:
            pattern:     Regex pattern to search for.
            path:        Directory or file to search within.
            include:     Glob filter, e.g. "*.py" for Python files only.
            max_results: Maximum number of matches to return.
        """
        full = self._safe_path(path)
        if include == "":
            include = None

        # Try system grep first (faster than Python)
        result = self._grep_system(pattern, full, include, max_results)
        if result is not None:
            return result

        # Python fallback (always works)
        return self._grep_python(pattern, full, include, max_results)

    def _grep_system(self, pattern: str, search_path: str,
                     include: Optional[str], max_results: int) -> Optional[str]:
        """System grep. Returns None if unavailable."""
        cmd = ["grep", "-rn", "--color=never", "-E"]
        if include:
            cmd.extend(["--include", include])
        for d in self._SKIP_DIRS:
            cmd.extend(["--exclude-dir", d])
        cmd.extend(["--exclude-dir", ".*"])
        cmd.extend([pattern, search_path])

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if r.returncode not in (0, 1):
            return None

        if not r.stdout.strip():
            return "No matches found."

        lines = []
        for line in r.stdout.strip().splitlines():
            if len(lines) >= max_results:
                break
            try:
                abs_file, rest = line.split(":", 1)
                rel = os.path.relpath(abs_file, self.root)
                lines.append(f"{rel}:{rest}")
            except ValueError:
                lines.append(line)

        out = "\n".join(lines)
        if len(lines) >= max_results:
            out += f"\n... (truncated at {max_results} results)"
        return out

    def _grep_python(self, pattern: str, search_path: str,
                     include: Optional[str], max_results: int) -> str:
        """Pure Python fallback. Uses cached file index when possible."""
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex: {e}"

        # Use cached index for full-workspace searches
        if search_path == self.root:
            candidates = self._get_file_index()
        else:
            candidates = []
            for root_dir, dirs, files in os.walk(search_path):
                dirs[:] = [d for d in dirs if not self._skip_dir(d)]
                for f in sorted(files):
                    fp = os.path.join(root_dir, f)
                    if not self._is_binary(fp):
                        candidates.append(fp)

        results = []
        for fp in candidates:
            if len(results) >= max_results:
                break
            if include and not fnmatch.fnmatch(os.path.basename(fp), include):
                continue
            try:
                with open(fp, "r") as fh:
                    for i, line in enumerate(fh, 1):
                        if regex.search(line):
                            rel = os.path.relpath(fp, self.root)
                            results.append(f"{rel}:{i}: {line.rstrip()}")
                            if len(results) >= max_results:
                                break
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

        if not results:
            return "No matches found."
        out = "\n".join(results)
        if len(results) >= max_results:
            out += f"\n... (truncated at {max_results} results)"
        return out

    # ── Tools: edit ──────────────────────────────────────────────────

    def edit_file(self, path: str, search_block: str, replace_block: str) -> str:
        """
        Find search_block in file, replace with replace_block.
        Previous content is saved under ``.agentfly_undo/`` for undo_edit.
        Rejects ambiguous edits (multiple matches).
        """
        full = self._safe_path(path)
        with open(full, "r") as f:
            content = f.read()

        if search_block not in content:
            return (
                "Error: search_block not found in file.\n"
                "Ensure whitespace and indentation match exactly.\n"
                "Use read_file to verify current content."
            )

        count = content.count(search_block)
        if count > 1:
            return (
                f"Error: search_block matches {count} locations.\n"
                "Include more surrounding context to make the match unique."
            )

        self._push_undo_snapshot(full, content)
        new_content = content.replace(search_block, replace_block, 1)
        with open(full, "w") as f:
            f.write(new_content)

        return self._edit_confirmation(full, new_content, replace_block)

    def create_file(self, path: str, content: str = "") -> str:
        """Create a new file. Fails if it already exists."""
        full = self._safe_path(path)
        if os.path.exists(full):
            return f"Error: {os.path.relpath(full, self.root)} already exists. Use edit_file."

        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)

        self._file_index_cache = None  # bust cache
        return f"Created {os.path.relpath(full, self.root)} ({len(content)} bytes)"

    def python(self, path: str, timeout: int = 60) -> str:
        """
        Run a Python file under the workspace: ``python3 <path>`` (fallback ``python``).
        Process cwd is the workspace root. Stdout and stderr are returned.
        """
        full = self._safe_path(path)
        if not os.path.isfile(full):
            return f"Error: Not a file or missing: {os.path.relpath(full, self.root)}"
        for exe in ("python3", "python"):
            try:
                r = subprocess.run(
                    [exe, full],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout),
                )
            except subprocess.TimeoutExpired:
                return f"Error: Python execution timed out after {timeout} seconds."
            except FileNotFoundError:
                continue
            parts: List[str] = []
            if r.stdout:
                parts.append(r.stdout.rstrip())
            if r.stderr:
                parts.append(r.stderr.rstrip())
            body = "\n".join(parts) if parts else "(no output)"
            if r.returncode != 0:
                return f"Exit code {r.returncode}\n{body}"
            return body
        return "Error: python3/python not found in PATH."

    # ── Tools: undo ──────────────────────────────────────────────────

    def undo_edit(self, path: str) -> str:
        """Revert file to its state before the last edit_file call (disk-backed stack)."""
        full = self._safe_path(path)
        previous = self._pop_undo_snapshot(full)
        if previous is None:
            return "Error: No edit history for this file."

        with open(full, "w") as f:
            f.write(previous)

        nums = self._list_undo_snapshot_names(self._undo_dir_for_file(full))
        n = len(nums)
        return f"Undo successful. ({n} earlier version{'s' if n != 1 else ''} in history)"

    def git_diff(self) -> str:
        """Show cumulative diff from the initial commit."""
        try:
            r = subprocess.run(
                ["git", "diff"], cwd=self.root,
                capture_output=True, text=True, timeout=10,
            )
            return r.stdout.strip() or "(no changes)"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "Error: git not available."

    def git_reset_file(self, path: str) -> str:
        """Hard-reset a single file to the initial commit."""
        full = self._safe_path(path)
        rel = os.path.relpath(full, self.root)
        try:
            subprocess.run(
                ["git", "checkout", "HEAD", "--", rel],
                cwd=self.root, capture_output=True, text=True,
                timeout=10, check=True,
            )
            self._clear_undo_for_file(full)
            return f"{rel} reset to initial commit."
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr.strip()}"
        except FileNotFoundError:
            return "Error: git not available."

    # ── Lifecycle ────────────────────────────────────────────────────

    def reset(self):
        """Remove on-disk undo data and clear caches. Call between episodes."""
        undo_root = os.path.join(self.root, self._UNDO_DIRNAME)
        if os.path.isdir(undo_root):
            shutil.rmtree(undo_root, ignore_errors=True)
        self._file_index_cache = None

    def history_stats(self) -> dict:
        """Introspection for debugging / trajectory logging."""
        files_with, total = self._undo_stats()
        return {
            "task_id": self.task_id,
            "workspace": self.root,
            "files_with_history": files_with,
            "total_snapshots": total,
        }

    # ── Private helpers ──────────────────────────────────────────────

    def _edit_confirmation(self, full_path: str, new_content: str,
                           replace_block: str) -> str:
        """Show a few lines of context around the edit."""
        rel = os.path.relpath(full_path, self.root)
        lines = new_content.splitlines(keepends=True)
        pos = new_content.find(replace_block)
        if pos == -1:
            return f"File updated: {rel}"

        line_start = new_content[:pos].count("\n")
        replace_lines = replace_block.count("\n") + 1
        ctx = 3
        s = max(0, line_start - ctx)
        e = min(len(lines), line_start + replace_lines + ctx)
        snippet = "".join(f"{i+1:6d} | {lines[i]}" for i in range(s, e))
        return f"File updated: {rel}\n[Lines {s+1}-{e}]\n{snippet}"


ALLOWED_TOOLS = frozenset({
    "read_file",
    "list_files",
    "grep_search",
    "edit_file",
    "create_file",
    "python",
    "undo_edit",
    "git_diff",
    "git_reset_file",
})


# ═════════════════════════════════════════════════════════════════════
#  CLI entry point (container / subprocess)
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        data = json.loads(sys.argv[1])
        task_id = data.get("task_id", "cli")
        tool_name = data["tool"]
        params = data.get("params", {})

        if tool_name not in ALLOWED_TOOLS:
            print(f"Error: Unknown tool '{tool_name}'", end="")
        else:
            fm = FileManager(task_id=task_id, workspace_root=data.get("workspace"))
            method = getattr(fm, tool_name, None)
            if method is None:
                print(f"Error: Tool '{tool_name}' not implemented", end="")
            else:
                try:
                    result = method(**params)
                except TypeError as e:
                    result = f"Error: Bad parameters for {tool_name}: {e}"
                except PermissionError as e:
                    result = str(e)
                except Exception as e:
                    result = f"Error: {type(e).__name__}: {e}"
                print(result, end="")
    except (IndexError, json.JSONDecodeError):
        print(
            'Usage: python3 file_manager.py '
            '\'{"task_id":"x","tool":"read_file","params":{"path":"main.py"}}\'',
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
