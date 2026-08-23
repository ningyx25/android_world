#!/bin/bash

# Emulator watchdog: exits (non-zero) when the QEMU emulator process dies,
# which in turn terminates the server (PID $SERVER_PID) and the container.
# Paired with `--restart unless-stopped` on the container, this converts a
# "QEMU is dead but the server serves 500s forever" outage (2026-08-23
# host-OOM incident) into an automatic full container restart, which is the
# cleanest recovery: emulator, a11y chain and wifi state are all rebuilt.
#
# Expected env:
#   SERVER_PID  PID of the python server process to terminate on QEMU death.

set -u

WATCH_INTERVAL_SEC="${WATCH_INTERVAL_SEC:-15}"
SERVER_PID="${SERVER_PID:?SERVER_PID must be set}"

# The bracketed pattern avoids matching this script's own /proc cmdline
# (pgrep -f matches full command lines, including shells holding this
# string as an argument).
QEMU_PATTERN="qemu-system-x86_64[-]headless"

echo "[watchdog] watching for QEMU (pattern: ${QEMU_PATTERN}) every ${WATCH_INTERVAL_SEC}s; server pid ${SERVER_PID}"

while true; do
  if ! pgrep -f "${QEMU_PATTERN}" >/dev/null 2>&1; then
    echo "[watchdog] QEMU process not found; killing server (pid ${SERVER_PID}) so the container exits and restarts." >&2
    kill "${SERVER_PID}" 2>/dev/null || true
    # Give the server a moment to shut down gracefully before the entrypoint
    # reaps us; the entrypoint exits non-zero either way.
    sleep 5
    exit 1
  fi
  sleep "${WATCH_INTERVAL_SEC}"
done
