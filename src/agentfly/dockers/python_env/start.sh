#!/bin/bash

# Listen on $PORT: AgentFly assigns each sandbox its own port (enroot containers share the host network).
PORT=${PORT:-8000}

echo "Starting Python HTTP Server on port $PORT"
exec uvicorn python_http_server:app --host 0.0.0.0 --port "$PORT" --workers 1 --loop asyncio
