
import subprocess
import json

def call_container_tool(container_id, tool_name, params):
    payload = json.dumps({"tool": tool_name, "params": params})
    # Use 'single quotes' for the JSON string to avoid shell escaping issues
    cmd = ["docker", "exec", container_id, "python3", "/tools/file_manager.py", payload]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return f"System Error: {result.stderr}"
    return result.stdout.strip()

# Example: Agent wants to read a file
# observation = call_container_tool("rl_env_1", "read_file", {"path": "app.py"})