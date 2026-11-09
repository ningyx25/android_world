#!/bin/bash

# Starts the emulator, then supervises the FastAPI server in the background.
#
# The server is NOT the container's PID 1 any more: it is a child of this
# supervisor, so `aw env restart` can replace it in place (seconds) without
# touching QEMU/adb. Only the watchdog's death (= QEMU gone) tears the
# container down, which lets `--restart unless-stopped` perform the full
# recreate -- preserving the 2026-08-23 OOM-incident recovery semantics.

set -u

./docker_setup/start_emu_headless.sh
EMU_BOOT_EXIT=$?
if [ ${EMU_BOOT_EXIT} -ne 0 ]; then
  echo "[entrypoint] emulator failed to boot (exit ${EMU_BOOT_EXIT}); exiting." >&2
  exit ${EMU_BOOT_EXIT}
fi

# A marker left behind by an interrupted emulator restart would keep the
# watchdog silent forever; clear it on every boot.
rm -f /tmp/.emulator_restart_in_progress

SERVER_LOG=/var/log/aw_server.log
mkdir -p /var/log
: > "${SERVER_LOG}"

python3 -m server.android_server >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
export SERVER_PID

./docker_setup/watchdog.sh &
WATCHDOG_PID=$!

cleanup() {
  kill "$WATCHDOG_PID" "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[entrypoint] server pid ${SERVER_PID}; watchdog pid ${WATCHDOG_PID}"

# Deliberately do NOT include SERVER_PID in the loop condition (unlike
# MobileWorld's supervisor): a crashed server is restarted here in place,
# and `restart` only has to kill it. The watchdog is the sole liveness gate.
while kill -0 "$WATCHDOG_PID" 2>/dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID" 2>/dev/null || true
    echo "[entrypoint] server exited; restarting it in place" >&2
    python3 -m server.android_server >> "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    export SERVER_PID
  fi
  sleep 2
done

echo "[entrypoint] watchdog exited (QEMU gone); exiting for full recreate" >&2
exit 1
