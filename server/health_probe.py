# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded, dependency-light health probes for the in-container emulator.

Ported from MobileWorld ``runtime/utils/health.py``. Every subprocess call is
timeout-boxed: a wedged adb must never hang a request or the Docker
HEALTHCHECK. This module deliberately uses only the standard library so it can
be imported by ``server.android_server`` without pulling in extra runtime
dependencies.
"""

import logging
import os
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

ADB_BIN = os.environ.get("ADB_BIN", "/opt/android/platform-tools/adb")
DEFAULT_DEVICE = os.environ.get("AW_DEVICE_SERIAL", "emulator-5554")
# Bracketed pattern avoids ``pgrep -f`` matching its own command line. Keep in
# sync with docker_setup/watchdog.sh's QEMU_PATTERN default.
QEMU_PATTERN = os.environ.get(
    "AW_QEMU_PATTERN", "qemu-system-x86_64[-]headless"
)
DEFAULT_PROBE_TIMEOUT = float(os.environ.get("ADB_PROBE_TIMEOUT_SECONDS", "3"))
QEMU_PROBE_TIMEOUT = float(os.environ.get("QEMU_PROBE_TIMEOUT_SECONDS", "2"))

# Machine-readable reasons, surfaced by /health and used to decide whether an
# adb-only soft recovery is worth attempting.
REASON_OK = "ok"
REASON_QEMU_MISSING = "qemu_missing"
REASON_ADB_TIMEOUT = "adb_timeout"
REASON_ADB_ERROR = "adb_error"
REASON_ADB_OFFLINE = "adb_offline"
REASON_ADB_UNAUTHORIZED = "adb_unauthorized"
REASON_DEVICE_MISSING = "device_missing"
REASON_BOOT_NOT_COMPLETED = "boot_not_completed"

# Reasons where QEMU is still alive and an adb reconnect/restart may help.
SOFT_RECOVERABLE_REASONS = frozenset({
    REASON_ADB_TIMEOUT,
    REASON_ADB_OFFLINE,
    REASON_ADB_UNAUTHORIZED,
    REASON_DEVICE_MISSING,
})


def _adb(
    args: list[str], timeout: float = DEFAULT_PROBE_TIMEOUT
) -> tuple[int | None, str]:
  """Runs adb with a hard timeout. Returns (returncode_or_None, stdout)."""
  cmd = [ADB_BIN] + args
  try:
    result = subprocess.run(  # pylint: disable=subprocess-run-check
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    return result.returncode, (result.stdout or "").strip()
  except (subprocess.TimeoutExpired, OSError) as exc:
    logger.debug("adb %s failed: %s", " ".join(cmd), exc)
    return None, ""


def qemu_process_alive(timeout: float = QEMU_PROBE_TIMEOUT) -> bool:
  """Returns True when a QEMU emulator process is running in this container."""
  try:
    result = subprocess.run(  # pylint: disable=subprocess-run-check
        ["pgrep", "-f", QEMU_PATTERN],
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode == 0:
      return True
  except (subprocess.TimeoutExpired, OSError):
    pass

  # Fallback: scan /proc/*/cmdline (works even when procps is unavailable).
  for pid in os.listdir("/proc"):
    if not pid.isdigit():
      continue
    try:
      with open(f"/proc/{pid}/cmdline", "rb") as handle:
        cmdline = handle.read().replace(b"\x00", b" ").decode(
            "utf-8", "replace"
        )
    except OSError:
      continue
    if "qemu-system-x86_64" in cmdline and "pgrep" not in cmdline:
      return True
  return False


def adb_device_state(
    device: str = DEFAULT_DEVICE,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> str:
  """Classifies ``adb get-state`` output into a machine-readable state."""
  returncode, output = _adb(["-s", device, "get-state"], timeout)
  if returncode is None:
    return "timeout"
  if output == "device":
    return "device"
  lowered = output.lower()
  if "offline" in lowered:
    return "offline"
  if "unauthorized" in lowered:
    return "unauthorized"
  if (
      "not found" in lowered
      or "no devices" in lowered
      or (returncode != 0 and not output)
  ):
    return "missing"
  return "error"


def adb_boot_completed(
    device: str = DEFAULT_DEVICE,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> bool:
  """True only when ``getprop sys.boot_completed`` returns exactly ``1``."""
  returncode, output = _adb(
      ["-s", device, "shell", "getprop", "sys.boot_completed"], timeout
  )
  return returncode == 0 and output.strip().strip("\r") == "1"


def adb_reconnect(
    device: str = DEFAULT_DEVICE,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> None:
  """Nudges a wedged adb connection back online (never touches QEMU)."""
  _adb(["reconnect", "offline"], timeout)
  _adb(["-s", device, "reconnect"], timeout)


def adb_restart_server(timeout: float = DEFAULT_PROBE_TIMEOUT) -> None:
  """Restarts the host-side adb server (never touches QEMU)."""
  _adb(["kill-server"], timeout)
  _adb(["start-server"], timeout)


def probe_device(
    device: str = DEFAULT_DEVICE,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
    include_qemu: bool = True,
) -> dict[str, Any]:
  """Returns a bounded, machine-readable health snapshot for one device."""
  qemu = qemu_process_alive() if include_qemu else True
  if not qemu:
    return {
        "device": device,
        "qemu": False,
        "adb_state": "unknown",
        "boot_completed": False,
        "healthy": False,
        "reason": REASON_QEMU_MISSING,
    }

  state = adb_device_state(device, timeout)
  if state != "device":
    reason = {
        "timeout": REASON_ADB_TIMEOUT,
        "offline": REASON_ADB_OFFLINE,
        "unauthorized": REASON_ADB_UNAUTHORIZED,
        "missing": REASON_DEVICE_MISSING,
    }.get(state, REASON_ADB_ERROR)
    return {
        "device": device,
        "qemu": True,
        "adb_state": state,
        "boot_completed": False,
        "healthy": False,
        "reason": reason,
    }

  boot = adb_boot_completed(device, timeout)
  return {
      "device": device,
      "qemu": True,
      "adb_state": state,
      "boot_completed": boot,
      "healthy": boot,
      "reason": REASON_OK if boot else REASON_BOOT_NOT_COMPLETED,
  }


def wait_for_device(
    device: str = DEFAULT_DEVICE,
    timeout: float = 30.0,
    interval: float = 1.0,
) -> bool:
  """Polls until adb reports the device in ``device`` state."""
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if adb_device_state(device) == "device":
      return True
    remaining = max(0.0, deadline - time.monotonic())
    time.sleep(min(interval, remaining))
  return adb_device_state(device) == "device"


def wait_for_boot(
    device: str = DEFAULT_DEVICE,
    timeout: float = 120.0,
    interval: float = 2.0,
    reconnect_every: float = 15.0,
) -> bool:
  """Polls until the device is booted, periodically nudging adb.

  Uses ``time.monotonic`` for the deadline and never sleeps past it.
  """
  deadline = time.monotonic() + timeout
  last_reconnect = time.monotonic()
  while time.monotonic() < deadline:
    if probe_device(device)["healthy"]:
      return True
    now = time.monotonic()
    if now - last_reconnect >= reconnect_every:
      adb_reconnect(device)
      last_reconnect = now
    remaining = max(0.0, deadline - time.monotonic())
    time.sleep(min(interval, remaining))
  return probe_device(device)["healthy"]


def soft_recovery(
    device: str = DEFAULT_DEVICE,
    timeout: float = 20.0,
) -> bool:
  """Cheap adb-only recovery: reconnect -> wait -> restart adb server -> wait.

  Never touches QEMU. A dead emulator is deliberately left to the watchdog,
  which triggers a full container recreate.
  """
  deadline = time.monotonic() + timeout
  adb_reconnect(device)
  remaining = max(0.0, deadline - time.monotonic())
  if wait_for_device(device, timeout=min(8.0, remaining)):
    remaining = max(0.0, deadline - time.monotonic())
    if wait_for_boot(device, timeout=remaining, interval=1.0):
      return True
  if time.monotonic() < deadline:
    adb_restart_server()
    remaining = max(0.0, deadline - time.monotonic())
    return wait_for_boot(device, timeout=remaining, interval=1.0)
  return False
