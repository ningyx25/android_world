# Credits to https://github.com/amrsa1/Android-Emulator-image

FROM eclipse-temurin:18-jdk-jammy

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /
#=============================
# Install Dependencies
#=============================
SHELL ["/bin/bash", "-c"]
RUN apt update && apt install -y curl sudo wget unzip bzip2 libdrm-dev libxkbcommon-dev libgbm-dev libasound-dev libnss3 libxcursor1 libpulse-dev libxshmfence-dev xauth xvfb x11vnc fluxbox wmctrl libdbus-glib-1-2 ffmpeg

#==============================
# Android SDK ARGS
#==============================
ARG ARCH="x86_64"
ARG TARGET="google_apis"
ARG API_LEVEL="33"
ARG BUILD_TOOLS="33.0.0"
ARG ANDROID_ARCH="x86_64"
ARG ANDROID_API_LEVEL="android-${API_LEVEL}"
ARG ANDROID_APIS="${TARGET};${ARCH}"
ARG EMULATOR_PACKAGE="system-images;${ANDROID_API_LEVEL};${ANDROID_APIS}"
ARG PLATFORM_VERSION="platforms;${ANDROID_API_LEVEL}"
ARG BUILD_TOOL="build-tools;${BUILD_TOOLS}"
ARG ANDROID_CMD="commandlinetools-linux-11076708_latest.zip"
ARG ANDROID_SDK_PACKAGES="${EMULATOR_PACKAGE} ${PLATFORM_VERSION} ${BUILD_TOOL} platform-tools emulator"

#==============================
# Set JAVA_HOME - SDK
#==============================
ENV ANDROID_SDK_ROOT=/opt/android
ENV PATH="$PATH:$ANDROID_SDK_ROOT/cmdline-tools/tools:$ANDROID_SDK_ROOT/cmdline-tools/tools/bin:$ANDROID_SDK_ROOT/emulator:$ANDROID_SDK_ROOT/tools/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/build-tools/${BUILD_TOOLS}"
ENV DOCKER="true"

#============================================
# Install required Android CMD-line tools
#============================================
RUN wget https://dl.google.com/android/repository/${ANDROID_CMD} -P /tmp && \
    unzip -d $ANDROID_SDK_ROOT /tmp/$ANDROID_CMD && \
    mkdir -p $ANDROID_SDK_ROOT/cmdline-tools/tools && cd $ANDROID_SDK_ROOT/cmdline-tools &&  mv NOTICE.txt source.properties bin lib tools/  && \
    cd $ANDROID_SDK_ROOT/cmdline-tools/tools && ls

#============================================
# Install required packages using SDK manager
#============================================
RUN yes Y | sdkmanager --licenses
RUN yes Y | sdkmanager --verbose --no_https ${ANDROID_SDK_PACKAGES}
#============================================
# Create required emulator
#============================================
ARG EMULATOR_NAME="Pixel_6_API_33"
ARG EMULATOR_DEVICE="pixel_6"
ENV EMULATOR_NAME=$EMULATOR_NAME
ENV DEVICE_NAME=$EMULATOR_DEVICE
RUN echo "no" | avdmanager --verbose create avd --force --name "${EMULATOR_NAME}" --device "${EMULATOR_DEVICE}" --package "${EMULATOR_PACKAGE}"


#====================================
# Install Python 3.11 from source
#====================================
RUN apt-get update && \
    apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev && \
    wget https://www.python.org/ftp/python/3.11.3/Python-3.11.3.tgz && \
    tar -xvf Python-3.11.3.tgz && \
    cd Python-3.11.3 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.11.3* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symlinks for python3.11
RUN ln -sf /usr/local/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.11 /usr/local/bin/python

# install pip
RUN apt-get update && apt-get install python3-pip -y

#====================================
# Install uv
#====================================
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ARG PATH="$PATH:/root/.local/bin"
ENV PATH="$PATH:/root/.local/bin"

#=========================
# Copy packaging metadata + proto sources FIRST, so editing server/*.py or the
# shell scripts does not invalidate the expensive dependency layer. setup.py's
# generate_protos cmdclass needs these files present at this point.
#=========================
COPY setup.py pyproject.toml README.md ./
COPY android_world/ /android_world/

#====================================
# Install dependencies (also generates *_pb2.py at image build time)
#====================================
RUN uv pip install . --system

#=========================
# Copy runtime code EXPLICITLY. A blanket `COPY . /` would also pull in the
# multi-GB baked AVD under docker_setup/prebaked/, duplicating it. The AVD is
# copied exactly once, directly into QEMU's AVD home, below.
#=========================
COPY apps/ /apps/
COPY assets/ /assets/
COPY docs/ /docs/
COPY scripts/ /scripts/
COPY server/ /server/
COPY .github/ /.github/
COPY docker_setup/*.sh /docker_setup/
COPY *.py *.md *.toml *.txt LICENSE .pylintrc .gitignore .dockerignore Dockerfile /

RUN chmod a+x /docker_setup/*.sh

#====================================
# Prebaked AVD: ONE copy, straight into /root/.android/avd -- the absolute
# path baked into hardware-qemu.ini and Pixel_6_API_33.ini, so the AVD must
# not be relocated afterwards. The directory-level COPY keeps the image
# buildable when prebaked/ contains only .gitkeep (no AVD yet); in that case
# the server falls back to runtime setup_apps via
# server/android_server.py:_verify_prebaked_snapshots.
# app_data and adbkey ride along in the same single copy; AW_APP_DATA_CACHE_DIR
# points at it directly, so no further cp -a of large data is needed.
#====================================
COPY docker_setup/prebaked/ /root/.android/avd/
RUN mkdir -p /root/.android && \
    if [ -f /root/.android/avd/adbkey ]; then \
      cp /root/.android/avd/adbkey /root/.android/avd/adbkey.pub /root/.android/ ; \
    fi

# Prebaked images skip the 24-app install loop at runtime. Build with
# --build-arg AW_PREBAKED=0 to keep the legacy cold-boot setup.
ARG AW_PREBAKED=1
ENV AW_PREBAKED=${AW_PREBAKED}
ENV AW_APP_DATA_CACHE_DIR=/root/.android/avd/app_data
ENV AW_DEVICE_SERIAL=emulator-5554

EXPOSE 5000

# Readiness signal for docker ps / external orchestration. Clients still poll
# GET /health directly (run_on_docker.py); they do not read this status.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

#=======================
# framework entry point
#=======================
ENTRYPOINT [ "./docker_setup/entrypoint.sh" ]