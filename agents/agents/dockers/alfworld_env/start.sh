#!/bin/bash
set -e

# ALFWorld HTTP Server Start Script
echo "=== ALFWorld HTTP Server Starting ==="

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate alfworld

# Set default config if not provided
CONFIG_FILE=${1:-"base_config.yaml"}
export ALFWORLD_CONFIG="/srv/${CONFIG_FILE}"

# Ensure config file exists
if [ ! -f "$ALFWORLD_CONFIG" ]; then
    echo "ERROR: Config file not found: $ALFWORLD_CONFIG"
    exit 1
fi

echo "Using config: $ALFWORLD_CONFIG"

# Verify ALFWorld data exists (should be included in image)
if [ ! -d "/root/.cache/alfworld/json_2.1.1" ]; then
    echo "WARNING: ALFWorld data not found, downloading..."
    alfworld-download
    echo "ALFWorld data downloaded successfully"
else
    echo "ALFWorld data found in image"
fi

# Set environment variables
export PORT=${PORT:-8000}
export PYTHONPATH="/srv:$PYTHONPATH"

# Start the server
echo "Starting ALFWorld HTTP server on port $PORT..."
cd /srv
exec python alfworld_http_server.py "$CONFIG_FILE" 