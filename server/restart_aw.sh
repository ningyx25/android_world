#!/bin/bash

# Restart the android_world server inside an existing container.
# Usage: ./restart_aw.sh <port> [--recreate] [image]
#
# Default: in-place restart. Kills the server process; entrypoint.sh's
# supervisor starts a fresh one. QEMU, adb and app data are untouched, so the
# environment is healthy again in seconds.
#
# --recreate: legacy stop -> rm -> run (full cold boot). Escape hatch for a
# wedged container, and the only path that re-resolves the image tag.

set -e

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <port> [--recreate] [image]"
    echo "Example: $0 5000"
    echo "         $0 5000 --recreate android_world:fast"
    exit 1
fi

PORT=$1
shift
RECREATE=0
IMAGE="${AW_IMAGE:-android_world:latest}"
for arg in "$@"; do
  case "$arg" in
    --recreate) RECREATE=1 ;;
    -*) echo "Unknown option: $arg" >&2; exit 1 ;;
    *) IMAGE="$arg" ;;
  esac
done

CONTAINER_NAME="aw_${PORT}"

if [ "$RECREATE" = "1" ]; then
  echo "Recreating container: ${CONTAINER_NAME} (image ${IMAGE})..."
  docker stop "${CONTAINER_NAME}" 2>/dev/null || true
  docker rm "${CONTAINER_NAME}" 2>/dev/null || true
  docker run -d --init --privileged \
      --name "${CONTAINER_NAME}" \
      -p "${PORT}:5000" \
      --restart unless-stopped \
      --memory 6g --memory-swap 6g \
      "${IMAGE}"
  echo "Done! Container ${CONTAINER_NAME} recreated on port ${PORT}"
  exit 0
fi

# In-place restart: the supervisor in entrypoint.sh notices the dead server
# and starts a fresh one. The bracketed pattern avoids pkill matching the
# `docker exec` command line itself.
echo "Restarting server in container: ${CONTAINER_NAME} (in place)..."
docker exec "${CONTAINER_NAME}" \
    pkill -f 'server[.]android_server' >/dev/null 2>&1 || true

HEALTH_TIMEOUT_SEC="${AW_HEALTH_TIMEOUT_SEC:-300}"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SEC ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if docker exec "${CONTAINER_NAME}" \
      curl -fsS http://localhost:5000/health >/dev/null 2>&1; then
    echo "Done! Server in ${CONTAINER_NAME} is healthy again (in-place restart)."
    exit 0
  fi
  sleep 2
done

echo "ERROR: server did not become healthy within ${HEALTH_TIMEOUT_SEC}s; see docker logs ${CONTAINER_NAME}" >&2
exit 1
