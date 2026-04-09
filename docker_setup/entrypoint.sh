#!/bin/bash

# Start Emulator
#============================================
./docker_setup/start_emu_headless.sh && \
adb -s "$(adb devices | grep emulator | head -1 | cut -f1)" root && \
python3 -m server.android_server
