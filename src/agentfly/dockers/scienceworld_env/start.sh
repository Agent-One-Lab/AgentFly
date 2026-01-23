#!/bin/bash

# Use PORT environment variable if set, otherwise default to 2700
PORT=${PORT:-2700}

echo "Starting Science World HTTP Server on port $PORT"
uvicorn scienceworld_server:app --host 0.0.0.0 --port $PORT --workers 1 --loop asyncio
