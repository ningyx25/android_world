#!/bin/bash

# Restart a Docker container for android_world
# Usage: ./restart_aw.sh <port>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <port>"
    echo "Example: $0 5000"
    exit 1
fi

PORT=$1
CONTAINER_NAME="aw_${PORT}"

echo "Stopping container: ${CONTAINER_NAME}..."
docker stop ${CONTAINER_NAME} 2>/dev/null || true

echo "Removing container: ${CONTAINER_NAME}..."
docker rm ${CONTAINER_NAME} 2>/dev/null || true

echo "Starting new container on port ${PORT}..."
docker run -d --init --privileged \
    --name ${CONTAINER_NAME} \
    -p ${PORT}:5000 \
    --restart unless-stopped \
    --memory 6g --memory-swap 6g \
    android_world:latest

echo "Done! Container ${CONTAINER_NAME} is running on port ${PORT}"
