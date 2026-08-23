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

"""Controller for Android that adds UI tree information to the observation."""

import contextlib
import enum
import hashlib
import io
import os
import time
from typing import Any
from typing import cast
from typing import Optional
from absl import logging
from android_env import env_interface
from android_env import loader
from android_env.components import config_classes
from android_env.proto.a11y import android_accessibility_forest_pb2
from android_env.wrappers import a11y_grpc_wrapper
from android_env.wrappers import base_wrapper
from android_world.env import adb_utils
from android_world.env import representation_utils
from android_world.utils import file_utils
import dm_env
import numpy as np
from PIL import Image


def _has_wrapper(
    env: env_interface.AndroidEnvInterface,
    target_wrapper: Any,
) -> bool:
  """Checks recursively if an environment object has a certain wrapper.

  Args:
    env: The environment object potentially wrapped.
    target_wrapper: The wrapper type to search for.

  Returns:
    True if the target_wrapper is found, otherwise False.
  """
  if isinstance(env, target_wrapper):
    return True
  elif hasattr(env, '_env'):
    return _has_wrapper(env._env, target_wrapper)  # pylint: disable=protected-access
  else:
    return False


def _unwrap_to_base_env(env: env_interface.AndroidEnvInterface):
  """Strips wrapper layers (e.g. A11yGrpcWrapper) down to the base AndroidEnv."""
  while hasattr(env, '_env'):
    env = env._env  # pylint: disable=protected-access
  return env


def get_a11y_tree(
    env: env_interface.AndroidEnvInterface,
    max_retries: int = 3,
    sleep_duration: float = 1.0,
) -> android_accessibility_forest_pb2.AndroidAccessibilityForest:
  """Gets a11y tree.

  Args:
    env: AndroidEnv.
    max_retries: Maximum number of retries to get a11y tree.
    sleep_duration: Time to sleep between each retry in seconds.

  Returns:
    A11y tree.

  Raises:
    RuntimeError: If the a11y tree was not able to be retrieved.
  """
  if not _has_wrapper(env, a11y_grpc_wrapper.A11yGrpcWrapper):
    raise ValueError(
        'Must use a11y_grpc_wrapper.A11yGrpcWrapper to get the a11y tree.'
    )
  env = cast(a11y_grpc_wrapper.A11yGrpcWrapper, env)
  if adb_utils.retry(3)(adb_utils.check_airplane_mode)(env):
    logging.warning(
        'Airplane mode is on -- cannot retrieve a11y tree via gRPC. Turning'
        ' it off...'
    )
    logging.info('Enabling networking...')
    env.attempt_enable_networking()
    time.sleep(1.0)

  forest: Optional[
      android_accessibility_forest_pb2.AndroidAccessibilityForest
  ] = None
  for _ in range(max_retries):
    try:
      forest = env.accumulate_new_extras()['accessibility_tree'][-1]  # pytype:disable=attribute-error
      return forest
    except KeyError:
      logging.warning('Could not get a11y tree, retrying.')
    time.sleep(sleep_duration)

  if forest is None:
    raise RuntimeError('Could not get a11y tree.')
  return forest


class A11yTreeUnavailableError(RuntimeError):
  """Raised when a11y tree is permanently unavailable after retries."""
  pass


_TASK_PATH = file_utils.convert_to_posix_path(
    file_utils.get_local_tmp_directory(), 'default.textproto'
)
DEFAULT_ADB_PATH = '~/Android/Sdk/platform-tools/adb'


# UI tree-specific keys that are added to observations:

# The forest is essentially a comprehensive snapshot of all user interface
# elements currently displayed on an Android device's screen. Each 'tree' in
# this 'forest' represents the accessibility details of a different window or
# screen section, providing structured information. The tree's origin is from
# the AccessibilityService. Please see the following for more detail:
# https://developer.android.com/reference/android/accessibilityservice/AccessibilityService

OBSERVATION_KEY_FOREST = 'forest'
# UI elements are specific nodes extracted from forest. See
# representation_utils.forest_to_ui_elements for details.
OBSERVATION_KEY_UI_ELEMENTS = 'ui_elements'


class A11yMethod(enum.Enum):
  """Method to get a11y tree."""

  # Custom gRPC wrapper that uses a11y forwarder app.
  A11Y_FORWARDER_APP = 'a11y_forwarder_app'

  # From `uiautomator dump``.
  UIAUTOMATOR = 'uiautomator'

  # No A11y tree retrieval
  NONE = 'none'


def apply_a11y_forwarder_app_wrapper(
    env: env_interface.AndroidEnvInterface, install_a11y_forwarding_app: bool
) -> env_interface.AndroidEnvInterface:
  return a11y_grpc_wrapper.A11yGrpcWrapper(
      env,
      install_a11y_forwarding=install_a11y_forwarding_app,
      start_a11y_service=True,
      enable_a11y_tree_info=True,
      latest_a11y_info_only=True,
  )


class AndroidWorldController(base_wrapper.BaseWrapper):
  """Controller for an Android instance that adds accessibility tree data.

  The Accessibility Tree in Android is a tree-based structure, originally for
  for assisting accessibility services. It provides information about UI
  elements (like text, buttons, and images) in a hierarchical format. The tree
  includes details such as the properties and actions available for each
  element.
  """

  # Number of consecutive a11y-forwarder failures before attempting to
  # re-bind the accessibility forwarder service on the device.
  _A11Y_FAILURES_BEFORE_REBIND = 3

  # Number of consecutive identical forests before the (silently stale)
  # accumulated tree is considered dead and recovery is triggered.
  _IDENTICAL_FOREST_STREAK_LIMIT = 20

  def __init__(
      self,
      env: env_interface.AndroidEnvInterface,
      a11y_method: A11yMethod = A11yMethod.A11Y_FORWARDER_APP,
      install_a11y_forwarding_app: bool = True,
  ):
    self._original_env = env
    if a11y_method == A11yMethod.A11Y_FORWARDER_APP:
      self._env = apply_a11y_forwarder_app_wrapper(
          env, install_a11y_forwarding_app
      )
      self._env.reset()  # Initializes required server services in a11y wrapper.
    else:
      self._env = env
    self._a11y_method = a11y_method
    self._a11y_failure_count = 0
    self._last_forest_hash = None
    self._identical_forest_streak = 0

  @property
  def device_screen_size(self) -> tuple[int, int]:
    """Returns the physical screen size of the device: (width, height)."""
    return adb_utils.get_screen_size(self._env)

  @property
  def logical_screen_size(self) -> tuple[int, int]:
    """Returns the logical screen size of the device.

    This will be different with the physical size if orientation or resolution
    is changed.
    """
    return adb_utils.get_logical_screen_size(self._env)

  @property
  def env(self) -> env_interface.AndroidEnvInterface:
    return self._env

  def refresh_env(self):
    # pylint: disable=protected-access
    # pytype: disable=attribute-error
    # Reconnect to emulator and reload a11y wrapper in case we lose connection.
    self._env = get_controller(
        console_port=self.env._coordinator._simulator._config.emulator_launcher.emulator_console_port,
        adb_path=self.env._coordinator._simulator._config.adb_controller.adb_path,
        grpc_port=self.env._coordinator._simulator._config.emulator_launcher.grpc_port,
    ).env
    # pylint: enable=protected-access
    # pytype: enable=attribute-error

  def get_pixels(self) -> np.ndarray:
    """Returns current screenshot pixels WITHOUT fetching the a11y tree.

    Two independent sources are tried in order; both yield the same pixels
    the timestep observation carries, but neither depends on the (fragile)
    a11y forwarding chain:
      1. The emulator's gRPC screenshot (the exact call the coordinator uses
         to populate timestep pixels).
      2. `adb exec-out screencap -p` (fully independent of android_env).

    Raises:
      RuntimeError: if both sources fail; callers may then fall back to the
        full get_state() path.
    """
    # 1) Emulator gRPC screenshot.
    try:
      base_env = _unwrap_to_base_env(self._env)
      pixels = np.asarray(
          # pylint: disable=protected-access
          base_env._coordinator._simulator.get_screenshot()
      )
      if pixels.size:
        return pixels.astype(np.uint8)
      raise RuntimeError('empty screenshot from emulator gRPC')
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('emulator gRPC screenshot failed: %s', e)
    # 2) adb screencap, independent of android_env internals.
    try:
      png_bytes = adb_utils.get_screenshot_png(self._env)
      return np.array(Image.open(io.BytesIO(png_bytes)), dtype=np.uint8)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('adb screencap fallback failed: %s', e)
    raise RuntimeError('all screenshot sources failed')

  # Stabilized-screenshot sampling parameters (see get_stable_pixels).
  _STABLE_PIXEL_ATTEMPTS = 6
  _STABLE_PIXEL_INTERVAL_SEC = 1.0

  def get_stable_pixels(
      self,
      max_attempts: int = _STABLE_PIXEL_ATTEMPTS,
      interval_sec: float = _STABLE_PIXEL_INTERVAL_SEC,
  ) -> np.ndarray:
    """Waits for the screen pixels to settle; a11y/uiautomator-free.

    Repeatedly captures pixels via get_pixels() (which never touches the
    a11y forwarding chain) and returns as soon as two consecutive captures
    are byte-identical. Video playback or looping animations never
    converge; on exhaustion the LAST capture is returned instead of
    failing -- a possibly-moving screenshot still beats a 500 (the old
    get_state-based stabilization path hard-failed exactly here, e.g. with
    VLC playing in the foreground).

    Args:
      max_attempts: Maximum number of captures (worst-case wall time is
        roughly max_attempts * (capture_time + interval_sec)).
      interval_sec: Sleep between captures.

    Returns:
      The most recent capture (stable if one was observed).

    Raises:
      RuntimeError: if even a single capture cannot be obtained (e.g. the
        emulator process is dead) -- there are no pixels to return.
    """
    prev_hash = None
    pixels = self.get_pixels()  # First capture outside the loop: fail fast.
    for _ in range(max_attempts - 1):
      new_hash = hashlib.md5(pixels.tobytes()).hexdigest()
      if new_hash == prev_hash:
        return pixels  # Two consecutive identical frames -> stable.
      prev_hash = new_hash
      time.sleep(interval_sec)
      pixels = self.get_pixels()
    return pixels  # Never stabilized; return the latest frame anyway.

  def _get_a11y_forest(
      self,
  ) -> android_accessibility_forest_pb2.AndroidAccessibilityForest:
    return get_a11y_tree(self._env)

  def get_a11y_forest(
      self,
  ) -> android_accessibility_forest_pb2.AndroidAccessibilityForest:
    """Returns the most recent a11y forest from the device."""
    try:
      return self._get_a11y_forest()
    # except RuntimeError:
    #   print(
    #       'Could not get a11y tree. Reconnecting to Android, reinitializing'
    #       ' AndroidEnv, and restarting a11y forwarding.'
    #   )
    #   self.refresh_env()
    #   try:
    #     return self._get_a11y_forest()
    except RuntimeError as e:
      raise A11yTreeUnavailableError('Could not get a11y tree after reconnect.') from e

  _FORWARDER_SERVICE = (
      'com.google.androidenv.accessibilityforwarder/'
      'com.google.androidenv.accessibilityforwarder.AccessibilityForwarder'
  )

  # The emulator's default virtual access-point SSID. Used by the wifi
  # self-heal ladder when an explicit reconnect is required. Overridable
  # via the AW_RECOVERY_WIFI_SSID environment variable.
  _WIFI_SSID = os.environ.get('AW_RECOVERY_WIFI_SSID', 'AndroidWifi')

  def _wlan0_routes_empty(self) -> bool:
    """Returns True if the device's wlan0 routing table has no entries.

    When the framework-level wifi breaks, app traffic has no route to the
    host (10.0.2.2), so the a11y forwarder can never reach the wrapper's
    gRPC server even though the service itself is healthy.
    """
    try:
      response = adb_utils.issue_generic_request(
          ['shell', 'ip', 'route', 'show', 'table', 'wlan0'], self._env,
      )
      routes = response.generic.output.decode('utf-8', errors='replace')
      return not routes.strip()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('wlan0 route check failed: %s', e)
      return True  # Assume broken; the repair attempts are cheap to skip.

  def _restore_device_network(self) -> None:
    """Escalating wifi recovery ladder with verification at each step.

    A bare `svc wifi disable/enable` toggle does NOT recover from every
    breakage: if the framework has blacklisted the SSID (logcat shows
    "Ignoring network selection disabled SSID: AndroidWifi"), wlan0 stays
    NO-CARRIER after the toggle. In that state an explicit
    `cmd wifi connect-network` is required (verified live, 2026-08-23).
    """
    try:
      if not self._wlan0_routes_empty():
        return
      logging.warning(
          'wlan0 routing table empty; toggling framework wifi.'
      )
      adb_utils.issue_generic_request(
          ['shell', 'svc', 'wifi', 'disable'], self._env,
      )
      time.sleep(3.0)
      adb_utils.issue_generic_request(
          ['shell', 'svc', 'wifi', 'enable'], self._env,
      )
      time.sleep(8.0)
      if not self._wlan0_routes_empty():
        return
      # The toggle did not restore routes (SSID likely blacklisted).
      # Explicitly reconnect to the emulator's virtual access point.
      logging.warning(
          'svc wifi toggle did not restore wlan0 routes; reconnecting'
          ' explicitly to SSID %s.',
          self._WIFI_SSID,
      )
      adb_utils.issue_generic_request(
          [
              'shell',
              'cmd',
              'wifi',
              'connect-network',
              self._WIFI_SSID,
              'open',
          ],
          self._env,
      )
      # Poll briefly for DHCP/route re-provisioning.
      for _ in range(3):
        time.sleep(3.0)
        if not self._wlan0_routes_empty():
          return
      logging.warning(
          'wlan0 routes still empty after connect-network; device network'
          ' may require container restart.'
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('wifi route check/repair failed: %s', e)

  def _try_rebind_a11y_forwarder(self) -> None:
    """Best-effort full recovery of a dead a11y forwarding chain.

    The chain breaks in three independent places (see
    DIAGNOSIS_http500_android_env.md, 2026-08-23), all of which must be
    repaired:
      1. The emulator's framework-level wifi can lose its routing table, so
         the device has no route to the host (10.0.2.2) and the forwarder
         can never reach the wrapper's gRPC server. Repair is a verified
         escalation ladder (see _restore_device_network): svc-wifi toggle,
         then explicit `cmd wifi connect-network` if the SSID was
         blacklisted.
      2. After the forwarder app crashes, Android parks it in "Crashed
         services" and does not re-bind it. Re-writing the *same* settings
         value does not trigger the settings observer -- the value must
         actually change (write null first).
      3. The restarted forwarder process loses the gRPC port and
         tree-logging flags that are delivered once at wrapper setup via
         one-shot broadcasts.
    """
    try:
      # 1) Restore the device->host network if the wlan0 routing table is
      #    empty (the framework re-provisions it on wifi re-enable).
      self._restore_device_network()

      # 2) Force a real settings change so the framework re-binds the
      #    crashed/unbound service.
      adb_utils.issue_generic_request(
          ['shell', 'settings', 'put', 'secure',
           'enabled_accessibility_services', 'null'],
          self._env,
      )
      time.sleep(1.0)
      adb_utils.issue_generic_request(
          ['shell', 'settings', 'put', 'secure',
           'enabled_accessibility_services', self._FORWARDER_SERVICE],
          self._env,
      )
      adb_utils.issue_generic_request(
          ['shell', 'settings', 'put', 'secure', 'accessibility_enabled', '1'],
          self._env,
      )
      time.sleep(2.0)

      # 3) Re-deliver the one-shot broadcasts the restarted process lost.
      # Private android_env 1.2.3 APIs (the version is pinned in
      # pyproject.toml); re-check on any android_env upgrade.
      self._env._configure_grpc()  # pylint: disable=protected-access
      self._env._enable_a11y_tree_logs()  # pylint: disable=protected-access
      logging.warning('a11y forwarder recovery sequence completed.')
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('a11y forwarder recovery attempt failed: %s', e)

  def _maybe_flag_stale_forest(
      self,
      forest: android_accessibility_forest_pb2.AndroidAccessibilityForest,
  ) -> None:
    """Detects a silently stale accumulated forest.

    The a11y wrapper's accumulated-extras buffer keeps returning the last
    forest it ever received even after the forwarder app dies mid-session,
    so an outdated tree looks exactly like success. A live screen almost
    always has at least one changing element (e.g. the status-bar clock), so
    many *identical* consecutive forests means the chain is dead.
    """
    forest_hash = hash(forest.SerializeToString())
    if forest_hash == self._last_forest_hash:
      self._identical_forest_streak += 1
    else:
      self._identical_forest_streak = 0
      self._last_forest_hash = forest_hash
    if self._identical_forest_streak >= self._IDENTICAL_FOREST_STREAK_LIMIT:
      logging.warning(
          'Identical a11y forest seen %d times in a row; the accumulated'
          ' tree is likely stale. Running a11y recovery.',
          self._identical_forest_streak,
      )
      self._identical_forest_streak = 0
      self._try_rebind_a11y_forwarder()

  def get_ui_elements(self) -> list[representation_utils.UIElement]:
    """Returns the most recent UI elements from the device."""
    if self._a11y_method == A11yMethod.A11Y_FORWARDER_APP:
      try:
        elements = representation_utils.forest_to_ui_elements(
            self.get_a11y_forest(),
            exclude_invisible_elements=True,
        )
        self._a11y_failure_count = 0
        return elements
      except A11yTreeUnavailableError:
        # Fall back to UIAUTOMATOR for THIS call only; keep preferring the
        # forwarder app on subsequent calls so the system can recover.
        self._a11y_failure_count += 1
        logging.warning(
            'a11y tree unavailable (failure %d), falling back to UIAUTOMATOR'
            ' for this call.', self._a11y_failure_count,
        )
        if self._a11y_failure_count % self._A11Y_FAILURES_BEFORE_REBIND == 0:
          self._try_rebind_a11y_forwarder()
        return representation_utils.xml_dump_to_ui_elements(
            adb_utils.uiautomator_dump(self._env)
        )
    elif self._a11y_method == A11yMethod.UIAUTOMATOR:
      return representation_utils.xml_dump_to_ui_elements(
          adb_utils.uiautomator_dump(self._env)
      )
    else:
      return []

  def _process_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Adds a11y tree info to the observation."""
    forest = None
    if self._a11y_method == A11yMethod.A11Y_FORWARDER_APP:
      try:
        forest = self.get_a11y_forest()
        ui_elements = representation_utils.forest_to_ui_elements(
            forest, exclude_invisible_elements=True
        )
        self._a11y_failure_count = 0
        self._maybe_flag_stale_forest(forest)
      except A11yTreeUnavailableError:
        # Fall back to UIAUTOMATOR for THIS call only; keep preferring the
        # forwarder app on subsequent calls so the system can recover.
        self._a11y_failure_count += 1
        logging.warning(
            'a11y tree unavailable in _process_timestep (failure %d), falling'
            ' back to UIAUTOMATOR for this call.', self._a11y_failure_count,
        )
        if self._a11y_failure_count % self._A11Y_FAILURES_BEFORE_REBIND == 0:
          self._try_rebind_a11y_forwarder()
        ui_elements = representation_utils.xml_dump_to_ui_elements(
            adb_utils.uiautomator_dump(self._env)
        )
    else:
      ui_elements = self.get_ui_elements()
    timestep.observation[OBSERVATION_KEY_FOREST] = forest
    timestep.observation[OBSERVATION_KEY_UI_ELEMENTS] = ui_elements
    return timestep

  def pull_file(
      self, remote_db_file_path: str, timeout_sec: Optional[float] = None
  ) -> contextlib._GeneratorContextManager[str]:
    """Pulls a file from the device to a temporary directory.

    The directory will be deleted when the context manager exits.
    Args:
      remote_db_file_path: The path to the file on the device.
      timeout_sec: Timeout in seconds for the adb calls.

    Returns:
      The path to the temporary directory containing the file.
    """
    remote_db_directory = os.path.dirname(remote_db_file_path)
    return file_utils.tmp_directory_from_device(
        remote_db_directory, self.env, timeout_sec
    )

  def push_file(
      self,
      local_db_file_path: str,
      remote_db_file_path: str,
      timeout_sec: Optional[float] = None,
  ) -> None:
    """Pushes a local file to the device."""

    remote_db_directory = os.path.dirname(remote_db_file_path)

    # First delete old .db, .db-wal, and .db-shm files.
    file_utils.clear_directory(remote_db_directory, self)
    file_utils.copy_data_to_device(
        local_db_file_path,
        remote_db_file_path,
        self.env,
        timeout_sec,
    )


def _write_default_task_proto() -> str:
  with open(_TASK_PATH, 'w') as f:
    f.write("""\
id: "default"

name: "Default task for device control."
description: "Empty task"

max_episode_sec: 7200  # Prevent infinite episodes.
  """)
  return _TASK_PATH


def get_controller(
    console_port: int = 5554,
    adb_path: str = DEFAULT_ADB_PATH,
    grpc_port: int = 8554,
) -> AndroidWorldController:
  """Creates a controller by connecting to an existing Android environment."""

  config = config_classes.AndroidEnvConfig(
      task=config_classes.FilesystemTaskConfig(
          path=_write_default_task_proto()
      ),
      simulator=config_classes.EmulatorConfig(
          emulator_launcher=config_classes.EmulatorLauncherConfig(
              emulator_console_port=console_port,
              adb_port=console_port + 1,
              grpc_port=grpc_port,
          ),
          adb_controller=config_classes.AdbControllerConfig(adb_path=adb_path),
      ),
  )
  android_env_instance = loader.load(config)
  logging.info('Setting up AndroidWorldController.')
  return AndroidWorldController(android_env_instance)
