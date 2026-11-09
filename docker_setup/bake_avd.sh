#!/bin/bash

# Bakes a fully initialized AVD (installed apps + app data snapshots + device
# settings) into docker_setup/prebaked/, so a runtime container can skip the
# 24-app setup_apps loop on every boot.
#
# !! NOT EXECUTED as part of the startup-speed change. Run this manually in a
# !! dedicated maintenance window on a KVM-capable host with GCS access.
# !! It creates and removes its OWN temporary container (name awbake_<ts>),
# !! never touches any aw_<port> / mobile_world_env_* container, never builds
# !! or retags android_world:latest, and never restarts a running service.
#
# How it works: the legacy image's server runs the complete setup_apps once
# during startup. We wait for /health, save a QEMU boot snapshot, export the
# AVD userdata (which now contains all apps + SNAPSHOT_DATA snapshots), and
# export the APK cache + adb keys as a fallback. No new server code is needed.

set -euo pipefail

IMAGE="${AW_BAKE_IMAGE:-android_world:latest}"
AVD_NAME="${EMULATOR_NAME:-Pixel_6_API_33}"
OUT_DIR="${AW_BAKE_OUT_DIR:-$(cd "$(dirname "$0")" && pwd)/prebaked}"
CONTAINER_NAME="awbake_$(date +%Y%m%d_%H%M%S)"
HEALTH_TIMEOUT="${AW_BAKE_TIMEOUT:-900}"

echo "[bake] image=${IMAGE} avd=${AVD_NAME} out=${OUT_DIR}"

# --- Preconditions (read-only) -------------------------------------------
if [ ! -e /dev/kvm ]; then
  echo "ERROR: /dev/kvm missing; baking requires a KVM-capable host." >&2
  exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: image ${IMAGE} not found locally." >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

cleanup() {
  echo "[bake] removing temporary container ${CONTAINER_NAME}"
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --- Start a private, un-restarted, port-less container -------------------
echo "[bake] starting temporary container ${CONTAINER_NAME}"
docker run -d --init --privileged --name "$CONTAINER_NAME" \
  --memory 6g --memory-swap 6g "$IMAGE"

# --- Wait for the legacy full setup to finish -----------------------------
echo "[bake] waiting for /health (up to ${HEALTH_TIMEOUT}s) -- setup_apps runs once here"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if docker exec "$CONTAINER_NAME" \
      curl -fsS http://localhost:5000/health >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
if ! docker exec "$CONTAINER_NAME" \
    curl -fsS http://localhost:5000/health >/dev/null 2>&1; then
  echo "ERROR: environment never became healthy; see docker logs ${CONTAINER_NAME}" >&2
  exit 1
fi

# --- Optional QEMU boot snapshot -----------------------------------------
if ! docker exec "$CONTAINER_NAME" adb emu avd snapshot save aw_baked >/dev/null 2>&1; then
  echo "[bake] WARNING: QEMU boot snapshot unsupported; the runtime will use the -no-snapshot cold-boot fallback" >&2
fi

# --- Export userdata, APK cache and adb keys ------------------------------
echo "[bake] exporting ${AVD_NAME}.avd and .ini"
docker cp "${CONTAINER_NAME}:/root/.android/avd/${AVD_NAME}.avd" "${OUT_DIR}/"
docker cp "${CONTAINER_NAME}:/root/.android/avd/${AVD_NAME}.ini" "${OUT_DIR}/"

# The APK cache is a fallback for the (rare) case where the server has to
# re-run setup_apps because the baked app snapshots are missing.
docker cp "${CONTAINER_NAME}:/tmp/android_world/app_data" "${OUT_DIR}/" 2>/dev/null || true
docker cp "${CONTAINER_NAME}:/root/.android/adbkey" "${OUT_DIR}/" 2>/dev/null || true
docker cp "${CONTAINER_NAME}:/root/.android/adbkey.pub" "${OUT_DIR}/" 2>/dev/null || true

# Reproducibility: commit only the script + this checksum, never the .avd.
(cd "$OUT_DIR" && find . -type f ! -name 'SHA256SUMS' ! -name '.gitkeep' \
  -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS) || true

echo "[bake] done. Next steps (in a separate approved window):"
echo "  docker build -t android_world:fast ."
echo "  uv run aw env run --image android_world:fast --port <port>"
echo "  # do NOT retag android_world:latest: restart_aw.sh re-resolves that tag."
