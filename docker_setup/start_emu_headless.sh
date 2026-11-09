#!/bin/bash

# Credits to https://github.com/amrsa1/Android-Emulator-image
#
# Boots the emulator, configures the guest, and hands a fully-rooted, booted
# device back to entrypoint.sh. Every adb/emulator probe is time-boxed: a
# wedged adb must never make EMULATOR_TIMEOUT meaningless (ported from
# MobileWorld docker/start_emulator.sh).

BL='\033[0;34m'
G='\033[0;32m'
RED='\033[0;31m'
YE='\033[1;33m'
NC='\033[0m' # No Color

emulator_name=${EMULATOR_NAME}
emulator_timeout=${EMULATOR_TIMEOUT:-300}
snapshot_timeout=${EMULATOR_SNAPSHOT_TIMEOUT:-120}
adb_root_timeout=${ADB_ROOT_TIMEOUT:-30}
adb_probe_timeout=10
boot_snapshot=${AW_BOOT_SNAPSHOT:-aw_baked}


function check_hardware_acceleration() {
    if [[ "${HW_ACCEL_OVERRIDE:-}" != "" ]]; then
        hw_accel_flag="$HW_ACCEL_OVERRIDE"
    elif timeout 20 emulator -accel-check >/dev/null 2>&1; then
        hw_accel_flag="-accel on"
    else
        hw_accel_flag="-accel off"
        echo "Warning: emulator -accel-check failed; KVM may be unavailable (expect high CPU)." >&2
    fi

    echo "$hw_accel_flag"
}


hw_accel_flag=$(check_hardware_acceleration)


function kill_stale_emulator() {
  # adb itself can be wedged when the guest went offline, so the adb path is
  # time-boxed and QEMU is hard-killed as a fallback.
  timeout 15 adb devices 2>/dev/null | grep emulator | cut -f1 | \
    xargs -r -I {} timeout 15 adb -s "{}" emu kill >/dev/null 2>&1 || true
  if pgrep -f '[q]emu-system-x86_64' >/dev/null 2>&1; then
    pkill -TERM -f '[q]emu-system-x86_64' 2>/dev/null || true
    for _ in $(seq 1 10); do
      pgrep -f '[q]emu-system-x86_64' >/dev/null 2>&1 || break
      sleep 1
    done
    pkill -KILL -f '[q]emu-system-x86_64' 2>/dev/null || true
  fi
}


function launch_emulator () {
  # $1 = 1 to boot from the baked QEMU snapshot, 0 for a full cold boot.
  local use_snapshot=$1

  # Generate adb keys if not present to avoid unauthorized error. Prebaked
  # images ship keys in /root/.android, so this is normally a no-op.
  if [ ! -f "$HOME/.android/adbkey" ]; then
    mkdir -p "$HOME/.android"
    timeout 15 adb keygen "$HOME/.android/adbkey" >/dev/null 2>&1 || true
  fi
  kill_stale_emulator
  timeout 15 adb start-server >/dev/null 2>&1 || true

  options="@${emulator_name} -no-window -no-boot-anim -memory 2048 ${hw_accel_flag} -grpc 8554"
  if [ "$use_snapshot" = "1" ]; then
    options="${options} -snapshot ${boot_snapshot}"
    echo "==> booting from QEMU snapshot '${boot_snapshot}'"
  else
    options="${options} -no-snapshot"
  fi

  if [[ "$OSTYPE" == *linux* ]]; then
    echo "${OSTYPE}: emulator ${options} -gpu off"
    nohup emulator $options -gpu off > /var/log/emulator.log 2>&1 &
  fi
  if [[ "$OSTYPE" == *darwin* ]] || [[ "$OSTYPE" == *macos* ]]; then
    echo "${OSTYPE}: emulator ${options} -gpu swiftshader_indirect"
    nohup emulator $options -gpu swiftshader_indirect > /var/log/emulator.log 2>&1 &
  fi
  return 0
}


function get_emulator_serial() {
  # Wait until an emulator appears with authorized 'device' status, but never
  # forever: the original loop had no deadline and could trap the container.
  local serial=""
  local deadline=$(( $(date +%s) + emulator_timeout ))
  while [ -z "$serial" ]; do
    serial=$(timeout "$adb_probe_timeout" adb devices 2>/dev/null | \
      grep emulator | grep -w 'device' | head -1 | cut -f1)
    if [ -z "$serial" ]; then
      if [ "$(date +%s)" -ge "$deadline" ]; then
        return 1
      fi
      sleep 1
    fi
  done
  echo "$serial"
}


function check_emulator_status () {
  local boot_timeout=${1:-$emulator_timeout}
  printf "${G}==> ${BL}Checking emulator booting up status 🧐${NC}\n"
  start_time=$(date +%s)
  spinner=( "⠹" "⠺" "⠼" "⠶" "⠦" "⠧" "⠇" "⠏" )
  i=0

  while true; do
    serial=$(get_emulator_serial)
    if [ -n "$serial" ]; then
      result=$(timeout "$adb_probe_timeout" adb -s "$serial" shell getprop sys.boot_completed 2>&1)
    else
      result=""
    fi

    if [ "$result" == "1" ]; then
      printf "\e[K${G}==> \u2713 Emulator is ready : '$result'           ${NC}\n"
      timeout "$adb_probe_timeout" adb devices -l >/dev/null 2>&1 || true
      timeout "$adb_probe_timeout" adb -s "$serial" shell input keyevent 82 >/dev/null 2>&1 || true
      return 0
    elif [ "$result" == "" ]; then
      printf "${YE}==> Emulator is partially Booted! 😕 ${spinner[$i]} ${NC}\r"
    else
      printf "${RED}==> $result, please wait ${spinner[$i]} ${NC}\r"
      i=$(( (i+1) % 8 ))
    fi

    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [ $elapsed_time -gt $boot_timeout ]; then
      printf "${RED}==> Timeout after ${boot_timeout} seconds elapsed 🕛.. ${NC}\n"
      return 1
    fi
    sleep 2
  done
}


function disable_animation() {
  serial=$(get_emulator_serial) || return 1
  # Three settings writes in one adb round-trip instead of three.
  timeout "$adb_probe_timeout" adb -s "$serial" shell \
    "settings put global window_animation_scale 0.0; settings put global transition_animation_scale 0.0; settings put global animator_duration_scale 0.0" \
    >/dev/null 2>&1 || true
}


function hidden_policy() {
  serial=$(get_emulator_serial) || return 1
  timeout "$adb_probe_timeout" adb -s "$serial" shell \
    "settings put global hidden_api_policy_pre_p_apps 1; settings put global hidden_api_policy_p_apps 1; settings put global hidden_api_policy 1" \
    >/dev/null 2>&1 || true
}


function configure_guest() {
  sleep 1
  disable_animation
  hidden_policy
  # adb root restarts adbd; wait for it before returning so the caller does
  # not hand a half-connected device to the server's first task.
  timeout 20 adb root >/dev/null 2>&1 || true
  root_deadline=$(( $(date +%s) + adb_root_timeout ))
  while [ "$(date +%s)" -lt "$root_deadline" ]; do
    if [ "$(timeout "$adb_probe_timeout" adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; then
      break
    fi
    sleep 1
  done
  echo "[start_emu_headless] emulator ready"
  return 0
}


# ---------------------------------------------------------------------------
# Boot sequence. A baked QEMU snapshot is optional; if it fails to come up we
# fall back to -no-snapshot from the same (prebaked) userdata, which still has
# all apps and app snapshots installed and only costs a 30-60s cold boot.
# ---------------------------------------------------------------------------
if [ "${AW_USE_BOOT_SNAPSHOT:-0}" = "1" ]; then
  launch_emulator 1
  if check_emulator_status "$snapshot_timeout"; then
    configure_guest
  else
    echo "==> snapshot boot failed; falling back to -no-snapshot cold boot" >&2
    kill_stale_emulator
    launch_emulator 0
    if ! check_emulator_status "$emulator_timeout"; then
      echo "Emulator failed to start properly, exiting..." >&2
      exit 1
    fi
    configure_guest
  fi
else
  launch_emulator 0
  if ! check_emulator_status "$emulator_timeout"; then
    echo "Emulator failed to start properly, exiting..." >&2
    exit 1
  fi
  configure_guest
fi

exit 0
