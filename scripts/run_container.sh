#!/usr/bin/env bash
# Starts an android_world environment container.
#
# Usage: ./scripts/run_container.sh [HOST_PORT]   (default 5000)
#
# --init runs an init process (PID 1) inside the container so orphaned
# processes (e.g. adb fork-servers left behind by crashed requests) get
# reaped instead of accumulating as zombies.
set -euo pipefail

PORT="${1:-5000}"
NAME="aw_${PORT}"

if ! docker image inspect android_world:latest >/dev/null 2>&1; then
  echo "android_world:latest not found; building it first (this can take a while)..."
  docker build -t android_world:latest "$(dirname "$0")/.."
fi

docker run -d --init --privileged \
  --name "${NAME}" \
  -p "${PORT}:5000" \
  android_world:latest

echo "Container ${NAME} started; environment will be ready at"
echo "http://localhost:${PORT}/health in ~5-10 minutes."
