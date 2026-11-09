#!/usr/bin/env bash
# Starts an android_world environment container.
#
# Usage: ./scripts/run_container.sh [HOST_PORT] [IMAGE]   (defaults 5000,
# android_world:latest). Set AW_IMAGE to override the image without a flag.
#
# --init runs an init process (PID 1) inside the container so orphaned
# processes (e.g. adb fork-servers left behind by crashed requests) get
# reaped instead of accumulating as zombies.
set -euo pipefail

PORT="${1:-5000}"
IMAGE="${2:-${AW_IMAGE:-android_world:latest}}"
NAME="aw_${PORT}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "${IMAGE} not found; building it first (this can take a while)..."
  docker build -t "${IMAGE}" "$(dirname "$0")/.."
fi

docker run -d --init --privileged \
  --name "${NAME}" \
  -p "${PORT}:5000" \
  --restart unless-stopped \
  --memory 6g --memory-swap 6g \
  "${IMAGE}"

echo "Container ${NAME} started; environment readiness is exposed at"
echo "http://localhost:${PORT}/health (prebaked images are ready in ~1-2 min;"
echo "legacy cold-boot images may take 5-10 minutes)."
