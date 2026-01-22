#!/bin/bash

# Use PORT environment variable if set, otherwise default to 8000
PORT=${PORT:-3000}

echo "Starting Webshop Environment HTTP Server on port $PORT"
uvicorn webshop_simulator_server:app --host 0.0.0.0 --port $PORT --workers 1 --loop asyncio
