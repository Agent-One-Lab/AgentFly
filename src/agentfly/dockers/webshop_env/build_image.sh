#!/bin/bash

# Exit on error
set -e

echo "Building webshop simulator Docker image..."

# Build the Docker image
docker build --no-cache --network=host -t webshop-simulator-env:latest .

echo "Build completed successfully!"
