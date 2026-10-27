# Code HTTP Server

The Code HTTP server provides a FastAPI-based execution environment for Python code snippets. It runs inside the sandbox container (enroot by default) and executes each snippet in a fresh process with time and memory limits.

The server is implemented as a FastAPI application that runs inside the sandbox container. The server script is located at `agentfly/dockers/python_env/python_http_server.py` and is designed to be executed within a containerized environment.

## API Endpoints

### Health Check

**GET** ``/health``

Check if the server is running and ready to accept requests.

**Response:**
- **200 OK**: Server is healthy
- **Content**: ``{"status": "ok"}``

### Execute Code

**POST** ``/exec``

Execute a Python code snippet in the sandbox environment.

**Request Body:**
- ``code`` (string): The Python code to execute

**Response:**
- **200 OK**: Code executed successfully
- **Content**: ``{"output": "..."}``
- **400 Bad Request**: Code execution failed or syntax error
- **408 Request Timeout**: Code execution timed out

## Implementation Details

The server uses a subprocess-based execution model:

1. **Process Spawning**: Each code execution creates a new Python subprocess
2. **Input Handling**: Code is passed via stdin to the child process
3. **Timeout Management**: Parent process enforces wall-clock timeouts
4. **Result Collection**: stdout/stderr are captured and returned
5. **Cleanup**: Process groups are killed on timeout or completion

As a result:

* Each execution starts from a clean interpreter; nothing carries over between requests
* Memory (1 GiB) and CPU time (8 s) are capped per execution, and wall time (10 s) by the parent
* Long-running or infinite loops are terminated
* Concurrent executions run in separate processes

This is resource control, not a security sandbox: see [Limits and isolation](index.md#limits-and-isolation).
