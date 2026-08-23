#!/bin/bash

# Start Emulator
#============================================
# Launches the emulator, then runs the FastAPI server in the foreground
# (via exec, so signals reach it directly) while a watchdog monitors the
# QEMU process in the background. If QEMU dies, the watchdog kills the
# server so this entrypoint exits non-zero; combined with
# `--restart unless-stopped` on the container this yields automatic
# recovery instead of indefinite 500s (2026-08-23 host-OOM incident).

set -u

./docker_setup/start_emu_headless.sh
EMU_BOOT_EXIT=$?
if [ ${EMU_BOOT_EXIT} -ne 0 ]; then
  echo "[entrypoint] emulator failed to boot (exit ${EMU_BOOT_EXIT}); exiting." >&2
  exit ${EMU_BOOT_EXIT}
fi

adb -s "$(adb devices | grep emulator | head -1 | cut -f1)" root
ADB_ROOT_EXIT=$?
if [ ${ADB_ROOT_EXIT} -ne 0 ]; then
  echo "[entrypoint] adb root failed (exit ${ADB_ROOT_EXIT}); continuing anyway." >&2
fi

# Watchdog: kills this script's python child (and thus the container) when
# QEMU disappears. Note the server is exec'd below, so its PID is $$ of
# this shell after exec -- captured here before exec.
SERVER_PID=$$
export SERVER_PID
./docker_setup/watchdog.sh &
WATCHDOG_PID=$!
trap 'kill ${WATCHDOG_PID} 2>/dev/null || true' EXIT

echo "[entrypoint] starting server (pid ${SERVER_PID}); watchdog pid ${WATCHDOG_PID}"
exec python3 -m server.android_server
