#!/bin/bash

# Exit on error
set -e

echo "Building scienceworld env docker image..."

# Build the Docker image
docker build --no-cache --network=host -t scienceworld-env:latest .

echo "Build completed successfully!"
