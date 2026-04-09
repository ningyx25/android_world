#!/bin/bash


# Credits to https://github.com/amrsa1/Android-Emulator-image

BL='\033[0;34m'
G='\033[0;32m'
RED='\033[0;31m'
YE='\033[1;33m'
NC='\033[0m' # No Color

emulator_name=${EMULATOR_NAME}

function check_hardware_acceleration() {
    if [[ "$HW_ACCEL_OVERRIDE" != "" ]]; then
        hw_accel_flag="$HW_ACCEL_OVERRIDE"
    else
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS-specific hardware acceleration check
            HW_ACCEL_SUPPORT=$(sysctl -a | grep -E -c '(vmx|svm)')
        else
            # generic Linux hardware acceleration check
            HW_ACCEL_SUPPORT=$(grep -E -c '(vmx|svm)' /proc/cpuinfo)
        fi

        if [[ $HW_ACCEL_SUPPORT == 0 ]]; then
            hw_accel_flag="-accel off"
            echo "Warning: no accelerator found. This Docker image is experimental and has only been tested on linux devices with KVM enabled."
        else
            hw_accel_flag="-accel on"
        fi
    fi

    echo "$hw_accel_flag"
}


hw_accel_flag=$(check_hardware_acceleration)

function launch_emulator () {
  # Generate adb keys if not present to avoid unauthorized error
  if [ ! -f "$HOME/.android/adbkey" ]; then
    mkdir -p "$HOME/.android"
    adb keygen "$HOME/.android/adbkey"
  fi
  adb kill-server
  adb start-server
  adb devices | grep emulator | cut -f1 | xargs -I {} adb -s "{}" emu kill
  # options="@${emulator_name} -no-window -no-snapshot -noaudio -no-boot-anim -memory 2048 ${hw_accel_flag} -camera-back none  -grpc 8554"
  options="@${emulator_name} -no-window -no-snapshot -no-boot-anim -memory 2048 ${hw_accel_flag} -grpc 8554"
  if [[ "$OSTYPE" == *linux* ]]; then
    echo "${OSTYPE}: emulator ${options} -gpu off"
    nohup emulator $options -gpu off &
  fi
  if [[ "$OSTYPE" == *darwin* ]] || [[ "$OSTYPE" == *macos* ]]; then
    echo "${OSTYPE}: emulator ${options} -gpu swiftshader_indirect"
    nohup emulator $options -gpu swiftshader_indirect &
  fi

  if [ $? -ne 0 ]; then
    echo "Error launching emulator"
    return 1
  fi
}


function get_emulator_serial() {
  # Wait until an emulator appears with authorized 'device' status
  local serial=""
  while [ -z "$serial" ]; do
    serial=$(adb devices | grep emulator | grep -w 'device' | head -1 | cut -f1)
    [ -z "$serial" ] && sleep 2
  done
  echo "$serial"
}

function check_emulator_status () {
  printf "${G}==> ${BL}Checking emulator booting up status 🧐${NC}\n"
  start_time=$(date +%s)
  spinner=( "⠹" "⠺" "⠼" "⠶" "⠦" "⠧" "⠇" "⠏" )
  i=0
  # Get the timeout value from the environment variable or use the default value of 300 seconds (5 minutes)
  timeout=${EMULATOR_TIMEOUT:-300}

  while true; do
    serial=$(get_emulator_serial)
    result=$(adb -s "$serial" shell getprop sys.boot_completed 2>&1)

    if [ "$result" == "1" ]; then
      printf "\e[K${G}==> \u2713 Emulator is ready : '$result'           ${NC}\n"
      adb devices -l
      adb -s "$serial" shell input keyevent 82
      return 0  # Return a 0 to indicate emulator has booted successfully
    elif [ "$result" == "" ]; then
      printf "${YE}==> Emulator is partially Booted! 😕 ${spinner[$i]} ${NC}\r"
    else
      printf "${RED}==> $result, please wait ${spinner[$i]} ${NC}\r"
      i=$(( (i+1) % 8 ))
    fi

    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [ $elapsed_time -gt $timeout ]; then
      printf "${RED}==> Timeout after ${timeout} seconds elapsed 🕛.. ${NC}\n"
      return 1 # Return a 1 to indicate failure if exceeded timeout
    fi
    sleep 4
  done
};


function disable_animation() {
  serial=$(get_emulator_serial)
  adb -s "$serial" shell "settings put global window_animation_scale 0.0"
  adb -s "$serial" shell "settings put global transition_animation_scale 0.0"
  adb -s "$serial" shell "settings put global animator_duration_scale 0.0"
};

function hidden_policy() {
  serial=$(get_emulator_serial)
  adb -s "$serial" shell "settings put global hidden_api_policy_pre_p_apps 1;settings put global hidden_api_policy_p_apps 1;settings put global hidden_api_policy 1"
};

launch_emulator
sleep 2

if check_emulator_status; then
  # Only run the below if the emulator is actually ready
  sleep 1
  disable_animation
  sleep 1
  hidden_policy
  sleep 1
else
  echo "Emulator failed to start properly, exiting..."
  exit 1
fi