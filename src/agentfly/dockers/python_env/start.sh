#!/bin/bash

# Use PORT environment variable if set, otherwise default to 8000
PORT=${PORT:-8000}

echo "Starting Python HTTP Server on port $PORT"
# uvicorn python_http_server:app --host 0.0.0.0 --port $PORT --workers 1 --loop asyncio
uvicorn python_http_server:app --host 0.0.0.0 --port 8000 --workers 1 --loop asyncio
