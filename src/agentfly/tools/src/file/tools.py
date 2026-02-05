from ...decorator import tool
from . import call_container_tool

@tool(name="read_file")
def read_file(path: str):
    """
    Reads a file from the workspace with line numbers.
    Args:
        path: The relative path to the file.
    """
    return call_container_tool("read_file", {"path": path})

@tool(name="edit_file")
def edit_file(path: str, search_block: str, replace_block: str):
    """
    Surgically replaces a block of text in a file. 
    Only replaces the first occurrence of search_block.
    Args:
        path: Path to file.
        search_block: Exact text to find.
        replace_block: Text to insert.
    """
    params = {
        "path": path, 
        "search_block": search_block, 
        "replace_block": replace_block
    }
    return call_container_tool("edit_file", params)

@tool(name="list_files")
def list_files(path: str = "."):
    """
    Lists all files recursively in the workspace.
    Args:
        path: Directory to start from (defaults to root).
    """
    return call_container_tool("list_files", {"path": path})

@tool(name="grep_search")
def grep_search(pattern: str, path: str = "."):
    """
    Search for a regex pattern across all files in a directory.
    Args:
        pattern: The regex/string to search for.
        path: Directory to search in.
    """
    return call_container_tool("grep_search", {"pattern": pattern, "path": path})

@tool(name="undo_edit")
def undo_edit(path: str):
    """
    Reverts the last modification made to a specific file.
    Args:
        path: The path of the file to revert.
    """
    return call_container_tool("undo_edit", {"path": path})