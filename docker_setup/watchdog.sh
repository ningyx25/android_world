#!/bin/bash

# Emulator watchdog: exits non-zero when the QEMU emulator process disappears,
# which makes entrypoint.sh exit and lets --restart unless-stopped perform a
# clean full-container recreate (emulator, a11y chain and wifi state rebuilt).
#
# Expected env:
#   SERVER_PID  PID of the python server process (a child of the supervisor
#               since entrypoint.sh no longer `exec`s it).
#
# Ported hardening from MobileWorld docker/watchdog.sh: a grace window of
# WATCH_MISS_LIMIT * WATCH_INTERVAL_SEC avoids killing the server on a
# transient pgrep miss, and an emulator-restart marker keeps the watchdog
# quiet while /health is deliberately restarting the emulator.

set -u

WATCH_INTERVAL_SEC="${WATCH_INTERVAL_SEC:-15}"
WATCH_MISS_LIMIT="${WATCH_MISS_LIMIT:-4}"
SERVER_PID="${SERVER_PID:?SERVER_PID must be set}"
QEMU_PATTERN="${QEMU_PATTERN:-qemu-system-x86_64[-]headless}"
MARKER="${EMULATOR_RESTART_MARKER:-/tmp/.emulator_restart_in_progress}"
MARKER_STALE_SEC="${EMULATOR_RESTART_STALE_SEC:-900}"

echo "[watchdog] watching QEMU (pattern: ${QEMU_PATTERN}) every ${WATCH_INTERVAL_SEC}s; server pid ${SERVER_PID}"

misses=0
while true; do
  if [ -f "$MARKER" ]; then
    mtime=$(stat -c %Y "$MARKER" 2>/dev/null || echo 0)
    if [ $(( $(date +%s) - mtime )) -lt "$MARKER_STALE_SEC" ]; then
      misses=0
      sleep "$WATCH_INTERVAL_SEC"
      continue
    fi
  fi

  if pgrep -f "$QEMU_PATTERN" >/dev/null 2>&1; then
    misses=0
  else
    misses=$((misses + 1))
    echo "[watchdog] QEMU missing (${misses}/${WATCH_MISS_LIMIT})" >&2
    if [ "$misses" -ge "$WATCH_MISS_LIMIT" ]; then
      echo "[watchdog] QEMU absent; killing server (pid ${SERVER_PID}) so the container exits and restarts." >&2
      kill -TERM "$SERVER_PID" 2>/dev/null || true
      # Catch a server that was re-parented after an in-place restart.
      pkill -TERM -f 'server[.]android_server' 2>/dev/null || true
      sleep 5
      exit 1
    fi
  fi
  sleep "$WATCH_INTERVAL_SEC"
done
