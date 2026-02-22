import os
import sys
import json
import shutil
import fnmatch
import re

class FileManager:
    def __init__(self, workspace_root=None):
        root = workspace_root or os.environ.get("WORKSPACE_ROOT", "/testbed")
        self.root = os.path.realpath(os.path.abspath(root))
        self.backup_dir = os.path.join(self.root, ".backup")
        os.makedirs(self.backup_dir, exist_ok=True)

    def _safe_path(self, path):
        # Treat absolute paths as relative to workspace root (e.g. "/" -> root, "/main.py" -> root/main.py)
        if path.startswith("/"):
            path = path[1:] or "."
        full_path = os.path.realpath(os.path.abspath(os.path.join(self.root, path)))
        if not full_path.startswith(self.root):
            raise PermissionError("Access denied: Path is outside workspace.")
        return full_path

    def _create_backup(self, path):
        """Saves a copy of the file before modification."""
        rel_path = os.path.relpath(path, self.root)
        backup_path = os.path.join(self.backup_dir, rel_path + ".bak")
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        if os.path.exists(path):
            shutil.copy2(path, backup_path)

    def read_file(self, path):
        full_path = self._safe_path(path)
        with open(full_path, 'r') as f:
            lines = f.readlines()
        return "".join([f"{i+1:4d} | {line}" for i, line in enumerate(lines)])

    def list_files(self, path="."):
        full_path = self._safe_path(path)
        files_list = []
        for root, _, files in os.walk(full_path):
            for f in files:
                rel_dir = os.path.relpath(root, self.root)
                files_list.append(os.path.join(rel_dir, f))
        return "\n".join(files_list)

    def grep_search(self, pattern, path="."):
        full_path = self._safe_path(path)
        results = []
        regex = re.compile(pattern)
        for root, _, files in os.walk(full_path):
            for f in files:
                f_path = os.path.join(root, f)
                try:
                    with open(f_path, 'r') as file:
                        for i, line in enumerate(file):
                            if regex.search(line):
                                results.append(f"{os.path.relpath(f_path, self.root)}:{i+1}: {line.strip()}")
                except UnicodeDecodeError: continue
        return "\n".join(results) if results else "No matches found."

    def edit_file(self, path, search_block, replace_block):
        full_path = self._safe_path(path)
        with open(full_path, 'r') as f:
            content = f.read()
        
        if search_block not in content:
            return "Error: Search block not found in file. Ensure exact matching."
        
        self._create_backup(full_path) # Backup before we change anything
        new_content = content.replace(search_block, replace_block, 1)
        
        with open(full_path, 'w') as f:
            f.write(new_content)
        return "File updated successfully."

    def undo_edit(self, path):
        full_path = self._safe_path(path)
        rel_path = os.path.relpath(full_path, self.root)
        backup_path = os.path.join(self.backup_dir, rel_path + ".bak")
        
        if not os.path.exists(backup_path):
            return "Error: No backup found for this file."
        
        shutil.move(backup_path, full_path)
        return "Last edit undone successfully."

if __name__ == "__main__":
    # Expecting: python3 file_manager.py '{"tool": "read_file", "params": {"path": "main.py"}}'
    try:
        input_data = json.loads(sys.argv[1])
        tool_name = input_data['tool']
        params = input_data.get('params', {})
        
        fm = FileManager()
        method = getattr(fm, tool_name)
        result = method(**params)
        print(result)
    except Exception as e:
        print(f"Execution Error: {str(e)}")
        sys.exit(1)