#!/bin/bash
set -e

# Docker Image Build Script for ALFWorld HTTP Environment
IMAGE_NAME="alfworld-http-env"
IMAGE_TAG="latest"

echo "=== Building ALFWorld HTTP Environment Docker Image ==="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Build the Docker image
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo ""
echo "=== Build Complete ==="
echo "Built image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Show image size
echo ""
echo "=== Image Details ==="
docker images | grep "${IMAGE_NAME}"

echo ""
echo "=== Build Successful! ==="
echo "To run: docker run -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"
