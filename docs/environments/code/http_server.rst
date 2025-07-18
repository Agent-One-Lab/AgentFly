.. _code_http_server:

Code HTTP Server
================

The Code HTTP server provides a FastAPI-based execution environment for Python code snippets. It runs inside Docker containers and offers secure, isolated code execution with timeout and resource controls.

Server Module Reference
-----------------------

.. currentmodule:: agents.dockers.python_env.python_http_server
.. automodule:: agents.dockers.python_env.python_http_server
    :members:
    :undoc-members:
    :show-inheritance:

API Endpoints
-------------

Health Check
~~~~~~~~~~~~

**GET** ``/health``

Check if the server is running and ready to accept requests.

**Response:**
   - **200 OK**: Server is healthy
   - **Content**: ``{"status": "ok"}``

Execute Code
~~~~~~~~~~~~

**POST** ``/exec``

Execute a Python code snippet in the sandbox environment.

**Request Body:**
   - ``code`` (string): The Python code to execute

**Response:**
   - **200 OK**: Code executed successfully
   - **Content**: ``{"output": "..."}``
   - **400 Bad Request**: Code execution failed or syntax error
   - **408 Request Timeout**: Code execution timed out

Server Configuration
--------------------

The server operates with the following built-in limits:

* **MAX_WALL**: 10 seconds wall-clock execution timeout
* **CHILD_MEM**: 1 GiB memory limit for child processes
* **CHILD_CPU**: 100% of a single CPU core
* **Output Truncation**: Output is limited to 16,384 characters
* **Error Truncation**: Error messages are limited to 8,192 characters

Security Model
--------------

The HTTP server implements a multi-layer security approach:

* **Process Isolation**: Code runs in separate child processes with ``os.setsid()``
* **Signal Handling**: Timeout enforcement via ``SIGKILL`` to process groups
* **Resource Limits**: Memory and CPU constraints on child processes
* **Output Sanitization**: Automatic truncation prevents memory exhaustion
* **Error Containment**: Exception handling prevents server crashes

Implementation Details
----------------------

The server uses a subprocess-based execution model:

1. **Process Spawning**: Each code execution creates a new Python subprocess
2. **Input Handling**: Code is passed via stdin to the child process
3. **Timeout Management**: Parent process enforces wall-clock timeouts
4. **Result Collection**: stdout/stderr are captured and returned
5. **Cleanup**: Process groups are killed on timeout or completion

The execution flow ensures that:

* No code can escape the sandbox environment
* Resource usage is strictly controlled
* Long-running or infinite loops are terminated
* Multiple concurrent executions are safely isolated 