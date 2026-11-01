#!/bin/bash

# Exit on error
set -e

echo "Building scienceworld env docker image..."

# Build the Docker image
# The name AgentFly's ScienceWorldSpec uses.
docker build --no-cache --network=host -t reasonwang/scienceworld-env:latest .

echo "Build completed successfully!"
