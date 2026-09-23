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

"""Environment interface for real-time interaction Android."""

import abc
import base64
import dataclasses
import io
import json
import time
from typing import Any, Optional, Self

from absl import logging
from android_env.components import action_type
from android_world.env import actuation
from android_world.env import adb_utils
from android_world.env import android_world_controller
from android_world.env import json_action
from android_world.env import representation_utils
import dm_env
import numpy as np
import pydantic
import requests
from PIL import Image


def _get_no_op_action() -> dict[str, Any]:
  """Creates a no-op action; used to retrieve screen & UI tree."""
  return {
      'action_type': np.array(action_type.ActionType.LIFT, dtype=np.int32),
      'touch_position': np.array((0.0, 0.0)),
  }


@dataclasses.dataclass(frozen=True)
class State:
  """State of the Android environment.

  Attributes:
    pixels: RGB array of current screen.
    forest: Raw UI forest; see android_world_controller.py for more info.
    ui_elements: Processed children and stateful UI elements extracted from
      forest.
    auxiliaries: Additional information about the state.
  """

  pixels: np.ndarray
  forest: Any
  ui_elements: list[representation_utils.UIElement]
  auxiliaries: dict[str, Any] | None = None

  @classmethod
  def create_and_infer_elements(
      cls,
      pixels: np.ndarray,
      forest: Any,
      screen_size: Optional[tuple[int, int]] = None,
  ) -> Self:
    """Creates a new instance, inferring UI elements from the forest."""

    elements = representation_utils.forest_to_ui_elements(
        forest, screen_size=screen_size
    )
    return cls(pixels, forest, elements)


class AsyncEnv(abc.ABC):
  """Interface for interacting with a real-time Android device.

  Computing environments, such as Android, run in real-time, independently of
  the agent interacting with it. All observations and actions are asynchronous
  and OS does not pause when providing observations or when accepting actions.
  Changes from action execution may take some time to appear.
  """

  @property
  @abc.abstractmethod
  def controller(self) -> android_world_controller.AndroidWorldController:
    """Returns the controller for the environment."""

  @abc.abstractmethod
  def reset(self, go_home: bool = False) -> State:
    """Go home on reset.

    Args:
      go_home: Whether to go home during the reset.
    """

  @abc.abstractmethod
  def get_state(self, wait_to_stabilize: bool = False) -> State:
    """Gets the state of the environment; i.e., screenshot & UI tree.

    In practice this will usually be called after executing an action. Logic
    should be implemented, perhaps a simple time.sleep, to ensure the
    environment updates after the action.

    Args:
      wait_to_stabilize: Whether to wait for the screen to stabilize before
        returning state.

    Returns:
      Observation containing RGB array of screen, the accessibility forest,
        and UI elements derived from the forest. See android_world_controller.py
        for
        more detail.
    """

  def display_message(self, message: str, header: str = '') -> None:
    """Displays a message on the screen."""

  @abc.abstractmethod
  def ask_question(
      self, question: str, timeout_seconds: float = -1.0
  ) -> str | None:
    """Asks a question to a hypothetical user in the environment.

    Common uses are to ask a question to clarify the user-provided goal, to ask
    for help when the agent is stuck, or when there is ambiguity in the current
    screen.

    Args:
      question: The question to ask the user.
      timeout_seconds: The timeout in seconds to wait for a response. If
        negative, then wait indefinitely.

    Returns:
      The response from the user or None if the user did not answer within the
      timeout.
    """

  @abc.abstractmethod
  def execute_action(self, action: json_action.JSONAction) -> None:
    """Executes action on the environment."""

  @property
  @abc.abstractmethod
  def foreground_activity_name(self) -> str:
    """Returns the activity name of the app currently opened in foreground."""

  @property
  @abc.abstractmethod
  def device_screen_size(self) -> tuple[int, int]:
    """Returns the screen size of the environment in pixels: (width, height)."""

  @property
  @abc.abstractmethod
  def logical_screen_size(self) -> tuple[int, int]:
    """Retrieves the logical screen size of the Android device.

    While the physical size is a fixed attribute of the display, the logical
    size is flexible and varies based on system settings such as the orientation
    or if the resolution is changed.

    Returns: The (width, height) in pixels, denoting the logical dimensions of
    the screen. Width and height values are aligned with the device's current
    orientation, meaning width is always logical horizontal direction (like in
    the landscape orientation width will be the physical vertical direction).
    """

  @abc.abstractmethod
  def close(self) -> None:
    """Closes the environment."""

  @property
  @abc.abstractmethod
  def interaction_cache(self) -> str:
    """Returns the interaction cache of the environment."""

  @abc.abstractmethod
  def hide_automation_ui(self) -> None:
    """Hides any UI, such as screen coordinates,."""

  @property
  @abc.abstractmethod
  def orientation(self) -> int:
    """Returns the orientation of the environment.

    Returns: 0 for portrait, 1 for landscape, 2 for reverse portrait,
    3 for reverse landscape.
    """

  @property
  @abc.abstractmethod
  def physical_frame_boundary(self) -> tuple[int, int, int, int]:
    """Returns the physical frame boundary of the environment.

    Returns: First two integers are the coordinates for top left corner, last
    two are for lower right corner. All coordinates are given in portrait
    orientation.
    """


def _process_timestep(timestep: dm_env.TimeStep) -> State:
  """Parses timestep observation and returns State."""
  return State(
      pixels=timestep.observation['pixels'],
      forest=timestep.observation[
          android_world_controller.OBSERVATION_KEY_FOREST
      ],
      ui_elements=timestep.observation[
          android_world_controller.OBSERVATION_KEY_UI_ELEMENTS
      ],
      auxiliaries={},
  )


class AsyncAndroidEnv(AsyncEnv):
  """Async environment interface using AndroidEnv to communicate with device."""

  interaction_cache = ''

  def __init__(
      self, controller: android_world_controller.AndroidWorldController
  ):
    self._controller = controller
    self._prior_state = None
    # Variable used to temporarily save interactions between agent and user.
    # Like when agent use answer action to answer user questions, we
    # use this to save the agent response. Or later on when agent has the
    # ability to ask user question, user's answer will be saved here as well.
    self.interaction_cache = ''

  @property
  def controller(self) -> android_world_controller.AndroidWorldController:
    return self._controller

  def reset(self, go_home: bool = False) -> State:
    if go_home:
      adb_utils.press_home_button(self.controller)
    self.interaction_cache = ''

    return _process_timestep(self.controller.reset())

  def _get_state(self):
    return _process_timestep(self.controller.step(_get_no_op_action()))

  def _get_stable_state(
      self,
      stability_threshold: int = 3,
      sleep_duration: float = 0.5,
      timeout: float = 6.0,
  ) -> State:
    """Checks if the UI elements remain stable over a number of checks and returns the state.

    Args:
        stability_threshold: Number of consecutive checks where UI elements must
          remain the same to consider UI stable.
        sleep_duration: Minimum time in seconds between each check.
        timeout: Maximum time in seconds to wait for UI to become stable before
          giving up.

    Returns:
        The current state of the UI if stability is achieved within the timeout.
    """
    if not self._prior_state:
      self._prior_state = self._get_state()
    if stability_threshold <= 0:
      raise ValueError('Stability threshold must be a positive integer.')

    stable_checks = 1
    start_time = time.time()
    deadline = start_time + timeout

    while stable_checks < stability_threshold and time.time() < deadline:
      iteration_start_time = time.time()
      current_state = self._get_state()

      if self._prior_state.ui_elements == current_state.ui_elements:
        stable_checks += 1
        if stable_checks == stability_threshold:
          break  # Exit early if stability is achieved.
      else:
        stable_checks = 1  # Reset if any change is detected
        self._prior_state = current_state

      elapsed_time = time.time() - iteration_start_time
      remaining_sleep = sleep_duration - elapsed_time
      if remaining_sleep > 0:
        sleep_time = min(remaining_sleep, deadline - time.time())
        if sleep_time > 0:
          time.sleep(sleep_time)
      # If remaining_sleep <= 0, proceed immediately to the next iteration

    return current_state  # pylint: disable=undefined-variable

  def get_state(self, wait_to_stabilize: bool = False) -> State:
    if wait_to_stabilize:
      return self._get_stable_state()
    return self._get_state()

  def execute_action(self, action: json_action.JSONAction) -> None:
    if action.action_type == json_action.ANSWER:
      self.interaction_cache = action.text
      if action.text:
        self.display_message(action.text, header='Agent answered:')
      return
    if action.action_type == json_action.STATUS:
      # Do nothing if it is a termination action.
      return
    state = self.get_state(wait_to_stabilize=False)
    actuation.execute_adb_action(
        action,
        state.ui_elements,
        self.logical_screen_size,
        self.controller,
    )

  def hide_automation_ui(self) -> None:
    """Hides the coordinates on screen."""
    adb_utils.issue_generic_request(
        'shell settings put system pointer_location 0', self.controller
    )

  def display_message(self, message: str, header: str = '') -> None:
    adb_utils.send_android_intent(
        command='broadcast',
        action='com.example.ACTION_UPDATE_OVERLAY',
        env=self.controller,
        extras={'task_type_string': header, 'goal_string': message},
    )

  def ask_question(
      self, question: str, timeout_seconds: float = -1.0
  ) -> str | None:
    raise NotImplementedError('ask_question is not implemented.')

  @property
  def foreground_activity_name(self) -> str:
    activity = adb_utils.get_current_activity(self.controller)[0]
    if activity:
      return activity
    else:
      return ''

  @property
  def device_screen_size(self) -> tuple[int, int]:
    return self.controller.device_screen_size

  @property
  def logical_screen_size(self) -> tuple[int, int]:
    return adb_utils.get_logical_screen_size(self.controller)

  def close(self) -> None:
    try:
      self.controller.close()
    except:  # pylint: disable=bare-except
      logging.warning('Failed to close controller. Continuing.')

  @property
  def orientation(self) -> int:
    return adb_utils.get_orientation(self.controller)

  @property
  def physical_frame_boundary(self) -> tuple[int, int, int, int]:
    return adb_utils.get_physical_frame_boundary(self.controller)


Params = dict[str, int | str]

# Default HTTP timeouts. Without these a hung server blocks the client
# forever; values must cover the server's worst-case internal retry chains
# (a degraded a11y path can take ~50s per state fetch).
DEFAULT_TIMEOUT_SEC = 60.0
SLOW_ENDPOINT_TIMEOUT_SEC = 120.0  # state fetching: score/teardown
INITIALIZE_TASK_TIMEOUT_SEC = 600.0  # task setup may install apps / download data
RESET_TIMEOUT_SEC = 300.0  # /reset may rebuild the whole controller
# Screenshot-specific timeout. Healthy baseline is ~0.05s; a dead-emulator
# server burns ~105s per attempt in internal retries before answering 500,
# so 45s deliberately cuts those off (the client-level retry ladder plus
# the episode-abort valve bounds the total damage instead).
SCREENSHOT_TIMEOUT_SEC = 45.0

def _post(url: str, **kwargs) -> requests.Response:
  """requests.post with a default timeout so calls can never hang forever."""
  kwargs.setdefault('timeout', DEFAULT_TIMEOUT_SEC)
  return requests.post(url, **kwargs)


def _get(url: str, **kwargs) -> requests.Response:
  """requests.get with a default timeout so calls can never hang forever."""
  kwargs.setdefault('timeout', DEFAULT_TIMEOUT_SEC)
  return requests.get(url, **kwargs)


class Response(pydantic.BaseModel):
  status: str
  message: str


class AndroidEnvClient:
  """Client for interacting with the Android environment server."""

  # HTTP statuses worth retrying for idempotent calls (transient env flakiness).
  RETRY_STATUSES = frozenset({500, 502, 503, 504})

  def __init__(self, base_url: str = "http://localhost:5000"):
    logging.info(
        "Setting up Android environment using Docker - Initial setup may take"
        " 5-10 minutes. Please wait..."
    )
    self.base_url = base_url

  def _request_with_retry(
      self,
      send_request,
      attempts: int = 3,
      backoff_s: float = 5.0,
      retry_statuses: frozenset[int] | None = RETRY_STATUSES,
  ):
    """Sends an HTTP request with retries on transient failures.

    Retries on connection-level errors and on `retry_statuses` responses
    (pass retry_statuses=frozenset() to disable HTTP-status retries, e.g. for
    non-idempotent action execution). Raises requests.HTTPError with the
    response body logged on final failure so root causes are visible
    client-side.
    """
    if retry_statuses is None:
      retry_statuses = self.RETRY_STATUSES
    last_response = None
    for attempt in range(1, attempts + 1):
      try:
        response = send_request()
      except (requests.ConnectionError, requests.Timeout) as e:
        last_response = None
        if attempt < attempts:
          logging.warning(
              "Request failed with connection error (%s); retry %d/%d in"
              " %.1fs", e, attempt, attempts - 1, backoff_s,
          )
          time.sleep(backoff_s)
          continue
        raise
      if response.status_code not in retry_statuses or attempt == attempts:
        if not response.ok:
          logging.error(
              "Request to %s failed with %d: %.500s",
              response.url, response.status_code, response.text,
          )
        response.raise_for_status()
        return response
      logging.warning(
          "Request to %s got %d; retry %d/%d in %.1fs",
          response.url, response.status_code, attempt, attempts - 1,
          backoff_s,
      )
      last_response = response
      time.sleep(backoff_s)
    # Unreachable, but keeps static analyzers happy.
    if last_response is not None:
      last_response.raise_for_status()
    raise RuntimeError("unreachable")

  def hide_automation_ui(self) -> None:
    """Hides the automation UI."""
    response = _post(f"{self.base_url}/hide_automation_ui")
    response.raise_for_status()
    return Response(**response.json())

  def reset(self, go_home: bool) -> Response:
    """Resets the environment."""
    response = self._request_with_retry(
        lambda: _post(
            f"{self.base_url}/reset",
            params={"go_home": go_home},
            timeout=RESET_TIMEOUT_SEC,
        )
    )
    return Response(**response.json())

  def get_screenshot(
      self, wait_to_stabilize: bool = False
  ) -> np.ndarray[Any, Any]:
    """Gets the current screenshot of the environment."""
    response = self._request_with_retry(
        lambda: _post(
            f"{self.base_url}/screenshot",
            params={"wait_to_stabilize": wait_to_stabilize},
            timeout=SCREENSHOT_TIMEOUT_SEC,
        )
    )
    image_bytes = base64.b64decode(response.json()["image_b64"])
    return np.array(Image.open(io.BytesIO(image_bytes)), dtype=np.uint8)
    # image = response.json()
    # return np.array(image["pixels"], dtype=np.uint8)

  def get_state(self, wait_to_stabilize: bool = False) -> State:
    """Gets the environment state: pixels plus the server-side UI elements.

    The UI elements come from the server's own `get_state()`, i.e. the exact
    list `/execute_action` indexes into, so element indices derived from this
    state are resolved identically server-side. There is deliberately no
    uiautomator/forest: in Docker mode the accessibility forest stays in the
    container, and clients that only need elements should use `ui_elements`
    (the `forest` field is None).

    Pixels round-trip through JPEG q=85, so they are close to but not
    bit-identical with the non-Docker environment; this matters for
    visualization only, not for element descriptions.

    Args:
      wait_to_stabilize: Whether to wait server-side for the screen to
        stabilize before returning (costs up to a few seconds, hence the
        longer timeout).

    Returns:
      State with pixels, forest=None, and the server-side UI elements.
    """
    response = self._request_with_retry(
        lambda: _post(
            f"{self.base_url}/state",
            params={"wait_to_stabilize": wait_to_stabilize},
            timeout=(
                SLOW_ENDPOINT_TIMEOUT_SEC
                if wait_to_stabilize
                else SCREENSHOT_TIMEOUT_SEC
            ),
        )
    )
    body = response.json()
    image_bytes = base64.b64decode(body["image_b64"])
    pixels = np.array(Image.open(io.BytesIO(image_bytes)), dtype=np.uint8)
    ui_elements = [
        representation_utils.ui_element_from_dict(element)
        for element in body["ui_elements"]
    ]
    return State(
        pixels=pixels,
        forest=None,
        ui_elements=ui_elements,
        auxiliaries={},
    )

  def execute_action(
      self,
      action: json_action.JSONAction,
  ) -> Response:
    """Executes an action in the environment."""
    print(f"Executing action: {action.json_str()}")
    # HTTP-level failures are NOT retried here: the action may already have
    # been executed server-side. Only connection-level errors (request never
    # reached the server) are retried via _request_with_retry.
    response = self._request_with_retry(
        lambda: _post(
            f"{self.base_url}/execute_action",
            json=json.loads(action.json_str()),
            timeout=SLOW_ENDPOINT_TIMEOUT_SEC,
        ),
        retry_statuses=frozenset(),
    )
    return Response(**response.json())

  def get_suite_task_list(self, max_index: int) -> list[str]:
    """Gets the list of tasks in the suite."""
    response = _get(
        f"{self.base_url}/suite/task_list", params={"max_index": max_index}
    )
    response.raise_for_status()
    return response.json()["task_list"]

  def get_suite_task_length(self, task_type: str) -> int:
    """Gets the length of the suite of tasks."""
    response = _get(
        f"{self.base_url}/suite/task_length", params={"task_type": task_type}
    )
    response.raise_for_status()
    return response.json()["length"]

  def reinitialize_suite(
      self,
      n_task_combinations: int = 2,  # Default from initial server setup.
      seed: int = 42,  # Default from initial server setup.
      task_family: str = "android_world",  # Default from initial server setup.
      use_identical_params: bool = False,
  ) -> Response:
    """Reinitializes the suite of tasks.

    Args:
      n_task_combinations: Number of task instances per task template.
      seed: Random seed for task parameter generation.
      task_family: Suite family to (re)build.
      use_identical_params: If True, all instances of a task share the same
        params (mirrors run.py's --fixed_task_seed).
    """
    response = _post(
        f"{self.base_url}/suite/reinitialize",
        params={
            "n_task_combinations": n_task_combinations,
            "seed": seed,
            "task_family": task_family,
            "use_identical_params": use_identical_params,
        },
        timeout=RESET_TIMEOUT_SEC,
    )
    response.raise_for_status()
    return Response(**response.json())

  def initialize_task(self, task_type: str, task_idx: int) -> Response:
    """Initializes the task in the environment."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = self._request_with_retry(
        lambda: _post(
            f"{self.base_url}/task/initialize",
            params=params,
            timeout=INITIALIZE_TASK_TIMEOUT_SEC,
        )
    )
    return Response(**response.json())

  def start_on_home_screen(self, task_type: str, task_idx: int) -> bool:
    """Gets whether the task starts on the home screen."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = self._request_with_retry(
        lambda: _post(
            f"{self.base_url}/task/start_on_home_screen", params=params
        )
    )
    return response.json()["start_on_home_screen"]

  def get_task_complexity(self, task_type: str, task_idx: int) -> int:
    """Gets the complexity of the current task."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = self._request_with_retry(
        lambda: _post(f"{self.base_url}/task/complexity", params=params)
    )
    return response.json()["complexity"]

  def tear_down_task(self, task_type: str, task_idx: int) -> Response:
    """Tears down the task in the environment."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = self._request_with_retry(
        lambda: _post(
            f"{self.base_url}/task/tear_down",
            params=params,
            timeout=SLOW_ENDPOINT_TIMEOUT_SEC,
        )
    )
    return Response(**response.json())

  def get_task_score(self, task_type: str, task_idx: int) -> float:
    """Gets the score of the current task."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = self._request_with_retry(
        lambda: _get(
            f"{self.base_url}/task/score",
            params=params,
            timeout=SLOW_ENDPOINT_TIMEOUT_SEC,
        )
    )
    return response.json()["score"]

  def get_task_goal(self, task_type: str, task_idx: int) -> str:
    """Gets the goal of the current task."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = self._request_with_retry(
        lambda: _get(f"{self.base_url}/task/goal", params=params)
    )
    return response.json()["goal"]

  def get_task_template(self, task_type: str, task_idx: int) -> str:
    """Gets the template of the current task."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = _get(f"{self.base_url}/task/template", params=params)
    response.raise_for_status()
    return response.json()["template"]

  def is_miniwob_episode_terminated(self) -> bool:
    """Checks if the MiniWoB episode is terminated."""
    response = _get(f"{self.base_url}/task/miniwob/is_epidode_terminated")
    response.raise_for_status()
    return response.json()["is_epidode_terminated"]

  def close(self) -> None:
    """Closes the environment."""
    response = _post(f"{self.base_url}/close")
    response.raise_for_status()

  def start_activity(
      self,
      activity: str,
      extra_args: list[str] | None = None,
      timeout_sec: float = 10,
  ) -> dict:
    """Launches the given activity."""
    params: dict = {"activity": activity, "timeout_sec": timeout_sec}
    if extra_args:
      params["extra_args"] = extra_args
    response = _post(f"{self.base_url}/adb/start_activity", params=params)
    response.raise_for_status()
    return response.json()

  def get_current_activity(self, timeout_sec: float = 10) -> str:
    """Returns the full activity name currently opened."""
    response = _get(f"{self.base_url}/adb/current_activity", params={"timeout_sec": timeout_sec})
    response.raise_for_status()
    return response.json()["activity"]

  def tap(self, x: int, y: int, timeout_sec: float = 10) -> None:
    """Taps the screen at (x, y)."""
    response = _post(f"{self.base_url}/adb/tap", params={"x": x, "y": y, "timeout_sec": timeout_sec})
    response.raise_for_status()

  def double_tap(self, x: int, y: int, timeout_sec: float = 10) -> None:
    """Double taps the screen at (x, y)."""
    response = _post(f"{self.base_url}/adb/double_tap", params={"x": x, "y": y, "timeout_sec": timeout_sec})
    response.raise_for_status()

  def long_press(self, x: int, y: int, timeout_sec: float = 10) -> None:
    """Long presses the screen at (x, y)."""
    response = _post(f"{self.base_url}/adb/long_press", params={"x": x, "y": y, "timeout_sec": timeout_sec})
    response.raise_for_status()

  def press_home_button(self, timeout_sec: float = 10) -> None:
    """Presses the HOME button."""
    response = _post(f"{self.base_url}/adb/press_home", params={"timeout_sec": timeout_sec})
    response.raise_for_status()

  def press_back_button(self, timeout_sec: float = 10) -> None:
    """Presses the BACK button."""
    response = _post(f"{self.base_url}/adb/press_back", params={"timeout_sec": timeout_sec})
    response.raise_for_status()

  def press_enter_button(self, timeout_sec: float = 10) -> None:
    """Presses the ENTER button."""
    response = _post(f"{self.base_url}/adb/press_enter", params={"timeout_sec": timeout_sec})
    response.raise_for_status()

  def press_key(self, keycode: str, timeout_sec: float = 10) -> None:
    """Presses any keyboard key by keycode."""
    response = _post(f"{self.base_url}/adb/press_key", params={"keycode": keycode, "timeout_sec": timeout_sec})
    response.raise_for_status()

  def type_text(self, text: str, timeout_sec: float = 10) -> None:
    """Types the specified text string."""
    response = _post(f"{self.base_url}/adb/type_text", params={"text": text, "timeout_sec": timeout_sec})
    response.raise_for_status()

  def get_adb_activity(self, app_name: str) -> str:
    """Gets the ADB activity for a given app name."""
    response = _get(f"{self.base_url}/adb/adb_activity", params={"app_name": app_name})
    response.raise_for_status()
    return response.json()["activity"]

  def get_all_package_names(self, timeout_sec: float = 10) -> list[str]:
    """Returns all installed package names."""
    response = _get(f"{self.base_url}/adb/all_packages", params={"timeout_sec": timeout_sec})
    response.raise_for_status()
    return response.json()["packages"]

  def get_all_apps(self, timeout_sec: float = 10) -> list[str]:
    """Returns all installed app names."""
    response = _get(f"{self.base_url}/adb/all_apps", params={"timeout_sec": timeout_sec})
    response.raise_for_status()
    return response.json()["apps"]

  def launch_app(self, app_name: str) -> bool:
    """Launches an app by name."""
    response = _post(f"{self.base_url}/adb/launch_app", params={"app_name": app_name})
    response.raise_for_status()
    return response.json()["launched"]

  def extract_package_name(self, activity: str) -> str:
    """Extracts the package name from an activity string."""
    response = _get(f"{self.base_url}/adb/extract_package_name", params={"activity": activity})
    response.raise_for_status()
    return response.json()["package_name"]

  def close_recents(self) -> None:
    """Closes all recent apps."""
    response = _post(f"{self.base_url}/adb/close_recents")
    response.raise_for_status()

  def close_app(self, app_name: str, timeout_sec: float = 10) -> bool:
    """Closes an app by name."""
    response = _post(f"{self.base_url}/adb/close_app", params={"app_name": app_name, "timeout_sec": timeout_sec})
    response.raise_for_status()
    return response.json()["closed"]

  def generate_swipe_command(
      self,
      start_x: int,
      start_y: int,
      end_x: int,
      end_y: int,
      duration_ms: int | None = None,
  ) -> list[str]:
    """Generates a swipe adb command argument list."""
    params: dict = {
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
    }
    if duration_ms is not None:
      params["duration_ms"] = duration_ms
    # Pure command generation (no side effects), safe to retry on HTTP 5xx.
    response = self._request_with_retry(
        lambda: _get(
            f"{self.base_url}/adb/generate_swipe_command", params=params
        )
    )
    response.raise_for_status()
    return response.json()["command"]

  def generate_drag_and_drop_command(
      self,
      start_x: int,
      start_y: int,
      end_x: int,
      end_y: int,
      duration_ms: int | None = None,
  ) -> list[str]:
    """Generates a drag and drop adb command argument list."""
    params: dict = {"start_x": start_x, "start_y": start_y, "end_x": end_x, "end_y": end_y}
    if duration_ms is not None:
      params["duration_ms"] = duration_ms
    response = _get(f"{self.base_url}/adb/generate_drag_and_drop_command", params=params)
    response.raise_for_status()
    return response.json()["command"]

  def send_intent(
      self,
      command: str,
      action: str,
      data_uri: str | None = None,
      mime_type: str | None = None,
      extras: dict | None = None,
      timeout_sec: int = 10,
  ) -> dict:
    """Sends an Android intent."""
    response = _post(
        f"{self.base_url}/adb/send_intent",
        json={"command": command, "action": action, "data_uri": data_uri,
              "mime_type": mime_type, "extras": extras, "timeout_sec": timeout_sec},
    )
    response.raise_for_status()
    return response.json()

  def get_api_level(self) -> int:
    """Gets the API level of the device."""
    response = _get(f"{self.base_url}/adb/api_level")
    response.raise_for_status()
    return response.json()["api_level"]

  def toggle_wifi(self, on_or_off: str) -> None:
    """Toggles wifi on or off."""
    response = _post(f"{self.base_url}/adb/toggle_wifi", params={"on_or_off": on_or_off})
    response.raise_for_status()

  def toggle_bluetooth(self, on_or_off: str) -> None:
    """Toggles Bluetooth on or off."""
    response = _post(f"{self.base_url}/adb/toggle_bluetooth", params={"on_or_off": on_or_off})
    response.raise_for_status()

  def set_brightness(self, max_or_min: str) -> None:
    """Sets screen brightness to max or min."""
    response = _post(f"{self.base_url}/adb/set_brightness", params={"max_or_min": max_or_min})
    response.raise_for_status()

  def clear_app_data(self, package_name: str) -> None:
    """Clears all data for a given package."""
    response = _post(f"{self.base_url}/adb/clear_app_data", params={"package_name": package_name})
    response.raise_for_status()

  def toggle_airplane_mode(self, on_or_off: str) -> None:
    """Toggles airplane mode on or off."""
    response = _post(f"{self.base_url}/adb/toggle_airplane_mode", params={"on_or_off": on_or_off})
    response.raise_for_status()

  def install_apk(self, apk_location: str) -> None:
    """Installs an APK."""
    response = _post(f"{self.base_url}/adb/install_apk", params={"apk_location": apk_location})
    response.raise_for_status()

  def check_airplane_mode(self) -> bool:
    """Checks if airplane mode is enabled."""
    response = _get(f"{self.base_url}/adb/airplane_mode")
    response.raise_for_status()
    return response.json()["enabled"]

  def extract_broadcast_data(self, raw_output: str) -> dict:
    """Extracts data from an adb broadcast command output."""
    response = _post(f"{self.base_url}/adb/extract_broadcast_data", params={"raw_output": raw_output})
    response.raise_for_status()
    return response.json()["data"]

  def get_clipboard_contents(self) -> str:
    """Gets the clipboard content."""
    response = _get(f"{self.base_url}/adb/clipboard")
    response.raise_for_status()
    return response.json()["content"]

  def change_orientation(self, orientation: str) -> None:
    """Changes the screen orientation."""
    response = _post(f"{self.base_url}/adb/change_orientation", params={"orientation": orientation})
    response.raise_for_status()

  def set_clipboard_contents(self, content: str) -> None:
    """Sets the clipboard content."""
    response = _post(f"{self.base_url}/adb/set_clipboard", params={"content": content})
    response.raise_for_status()

  def grant_permissions(self, activity_name: str, permission: str) -> None:
    """Grants a permission to an activity."""
    response = _post(f"{self.base_url}/adb/grant_permissions",
                             params={"activity_name": activity_name, "permission": permission})
    response.raise_for_status()

  def execute_sql_command(self, db_path: str, sql_command: str) -> dict:
    """Executes an SQL command on a SQLite database via ADB."""
    response = _post(f"{self.base_url}/adb/execute_sql",
                             params={"db_path": db_path, "sql_command": sql_command})
    response.raise_for_status()
    return response.json()

  def get_call_state(self, timeout_sec: float = 10) -> str:
    """Gets the current call state."""
    response = _get(f"{self.base_url}/adb/call_state", params={"timeout_sec": timeout_sec})
    response.raise_for_status()
    return response.json()["state"]

  def call_emulator(self, phone_number: str, timeout_sec: float = 10) -> None:
    """Simulates an incoming call in the emulator."""
    response = _post(f"{self.base_url}/adb/call_emulator",
                             params={"phone_number": phone_number, "timeout_sec": timeout_sec})
    response.raise_for_status()

  def end_call_if_active(self, timeout_sec: float = 10) -> None:
    """Ends the phone call if active."""
    response = _post(f"{self.base_url}/adb/end_call", params={"timeout_sec": timeout_sec})
    response.raise_for_status()

  def clear_android_emulator_call_log(self, timeout_sec: float = 10) -> None:
    """Clears the call log."""
    response = _post(f"{self.base_url}/adb/clear_call_log", params={"timeout_sec": timeout_sec})
    response.raise_for_status()

  def call_phone_number(self, phone_number: str, timeout_sec: float = 10) -> None:
    """Initiates a phone call."""
    response = _post(f"{self.base_url}/adb/call_phone",
                             params={"phone_number": phone_number, "timeout_sec": timeout_sec})
    response.raise_for_status()

  def text_emulator(self, phone_number: str, message: str, timeout_sec: float = 10) -> None:
    """Simulates an incoming SMS in the emulator."""
    response = _post(f"{self.base_url}/adb/text_emulator",
                             params={"phone_number": phone_number, "message": message, "timeout_sec": timeout_sec})
    response.raise_for_status()

  def set_default_app(self, setting_key: str, package_name: str, timeout_sec: float = 10) -> None:
    """Sets the default app for a given setting key."""
    response = _post(f"{self.base_url}/adb/set_default_app",
                             params={"setting_key": setting_key, "package_name": package_name, "timeout_sec": timeout_sec})
    response.raise_for_status()

  def disable_headsup_notifications(self, timeout_sec: float = 10) -> None:
    """Disables heads-up notifications."""
    response = _post(f"{self.base_url}/adb/disable_headsup", params={"timeout_sec": timeout_sec})
    response.raise_for_status()

  def enable_headsup_notifications(self, timeout_sec: float = 10) -> None:
    """Enables heads-up notifications."""
    response = _post(f"{self.base_url}/adb/enable_headsup", params={"timeout_sec": timeout_sec})
    response.raise_for_status()

  def put_settings(self, namespace: int, key: str, value: str) -> None:
    """Changes a system setting via ADB."""
    response = _post(f"{self.base_url}/adb/put_settings",
                             params={"namespace": namespace, "key": key, "value": value})
    response.raise_for_status()

  def get_all_settings(self) -> dict:
    """Gets all system settings."""
    response = _get(f"{self.base_url}/adb/all_settings")
    response.raise_for_status()
    return response.json()["settings"]

  def delete_contacts(self, timeout_sec: float = 10) -> None:
    """Deletes all contacts."""
    response = _post(f"{self.base_url}/adb/delete_contacts", params={"timeout_sec": timeout_sec})
    response.raise_for_status()

  def get_screen_size(self) -> tuple[int, int]:
    """Gets the physical screen size in pixels."""
    response = _get(f"{self.base_url}/adb/screen_size")
    response.raise_for_status()
    data = response.json()
    return data["width"], data["height"]

  def get_logical_screen_size(self) -> tuple[int, int]:
    """Gets the logical screen size."""
    response = _get(f"{self.base_url}/adb/logical_screen_size")
    response.raise_for_status()
    data = response.json()
    return data["width"], data["height"]

  def get_physical_frame_boundary(self) -> tuple[int, int, int, int]:
    """Gets the physical frame boundary."""
    response = _get(f"{self.base_url}/adb/physical_frame_boundary")
    response.raise_for_status()
    data = response.json()
    return data["x1"], data["y1"], data["x2"], data["y2"]

  def get_orientation(self) -> int:
    """Gets the current screen orientation."""
    response = _get(f"{self.base_url}/adb/orientation")
    response.raise_for_status()
    return response.json()["orientation"]

  def set_screen_size(self, width: int, height: int) -> None:
    """Sets the logical screen size."""
    response = _post(f"{self.base_url}/adb/set_screen_size",
                             params={"width": width, "height": height})
    response.raise_for_status()

  def retry(self, n: int, func_name: str) -> dict:
    """Retries an adb_utils function up to n times on AdbControllerError."""
    response = _post(f"{self.base_url}/adb/retry",
                             params={"n": n, "func_name": func_name})
    response.raise_for_status()
    return response.json()

  def set_root_if_needed(self, timeout_sec: float | None = None) -> None:
    """Sets ADB to root if not already."""
    params: dict = {}
    if timeout_sec is not None:
      params["timeout_sec"] = timeout_sec
    response = _post(f"{self.base_url}/adb/set_root", params=params)
    response.raise_for_status()

  def uiautomator_dump(self, timeout_sec: float = 30) -> str:
    """Returns the UI hierarchy via uiautomator dump."""
    response = _get(f"{self.base_url}/adb/uiautomator_dump", params={"timeout_sec": timeout_sec})
    response.raise_for_status()
    return response.json()["ui_hierarchy"]

  def issue_generic_request(
      self,
      args: list[str] | str,
      timeout_sec: float = 10,
  ) -> dict:
    """Issues a generic adb command."""
    # HTTP-level failures are NOT retried (the adb command may already have
    # run server-side); only connection-level errors are retried, matching
    # execute_action's policy.
    response = self._request_with_retry(
        lambda: _post(
            f"{self.base_url}/adb/generic_request",
            params={"timeout_sec": timeout_sec},
            json=args,
        ),
        retry_statuses=frozenset(),
    )
    response.raise_for_status()
    return response.json()

  def health(self) -> bool:
    """Checks the health of the environment."""
    try:
      response = _get(f"{self.base_url}/health")
      response.raise_for_status()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Environment is not healthy: {e}")
      return False
    return True
