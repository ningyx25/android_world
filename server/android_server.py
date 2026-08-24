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

"""FastAPI server for managing and interacting with an Android environment.

This server exposes endpoints to control an Android emulator, execute tasks,
and manage task execution on AndroidWorld tasks.
"""

import base64
import contextlib
import io
import logging
import subprocess
import time
import typing
from typing import Any, Optional

from PIL import Image

from android_world import registry as aw_registry_module
from android_world import suite_utils
from android_world.env import adb_utils
from android_world.env import env_launcher
from android_world.env import interface
from android_world.env import json_action
from android_world.task_evals.miniwob.miniwob_base import get_episode_reward

import fastapi
import pydantic
import uvicorn

logger = logging.getLogger(__name__)


class StateResponse(pydantic.BaseModel):
  """Pydantic model for state responses, including pixels and UI elements."""

  pixels: list[int]
  ui_elements: list[Any]


class SendIntentRequest(pydantic.BaseModel):
  command: str
  action: str
  data_uri: Optional[str] = None
  mime_type: Optional[str] = None
  extras: Optional[dict[str, Any]] = None
  timeout_sec: int = 10


@contextlib.asynccontextmanager
async def lifespan(fast_api_app: fastapi.FastAPI):
  """Manages the lifecycle of the Android environment and task suite."""
  adb_path = "/opt/android/platform-tools/adb"
  fast_api_app.state.adb_path = adb_path
  fast_api_app.state.app_android_env = env_launcher.load_and_setup_env(
      console_port=5554,
      emulator_setup=True,
      freeze_datetime=True,
      adb_path=adb_path,
  )
  task_registry = aw_registry_module.TaskRegistry()
  aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
  initial_suite = suite_utils.create_suite(
      task_registry=aw_registry,
      n_task_combinations=2,
      seed=42,  # Optional: for reproducibility
  )
  fast_api_app.state.suite = initial_suite
  fast_api_app.state.task_registry = task_registry
  yield
  # Shutdown
  if fast_api_app.state.app_android_env is not None:
    fast_api_app.state.app_android_env.close()


app = fastapi.FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: fastapi.Request, exc: Exception):
  """Returns 500 with a short exception summary instead of a bare body.

  The full traceback is still logged server-side; the summary lets HTTP
  clients (e.g. the eval runner) see the root cause without reading container
  logs.
  """
  logger.exception(
      "Unhandled error on %s %s", request.method, request.url.path
  )
  return fastapi.responses.JSONResponse(
      status_code=500,
      content={"detail": f"{type(exc).__name__}: {exc}"},
  )


suite_router = fastapi.APIRouter(prefix="/suite", tags=["suite"])
task_router = fastapi.APIRouter(prefix="/task", tags=["task"])
miniwob_router = fastapi.APIRouter(prefix="/miniwob", tags=["miniwob"])
adb_router = fastapi.APIRouter(prefix="/adb", tags=["adb"])


def get_app_android_env(request: fastapi.Request) -> interface.AsyncEnv:
  """Dependency to get the application's Android environment instance."""
  return request.app.state.app_android_env


def get_app_suite(request: fastapi.Request) -> suite_utils.Suite:
  """Dependency to get the application's task suite instance."""
  return request.app.state.suite


AndroidEnv = typing.Annotated[
    interface.AsyncEnv, fastapi.Depends(get_app_android_env)
]
AndroidSuite = typing.Annotated[
    suite_utils.Suite, fastapi.Depends(get_app_suite)
]


@app.post("/hide_automation_ui")
async def hide_automation_ui(app_android_env: AndroidEnv):
  """Hides the automation UI elements from the Android environment."""
  app_android_env.hide_automation_ui()
  return {"status": "success", "message": "Automation UI hidden."}


@app.post("/reset")
async def reset(go_home: bool, app_android_env: AndroidEnv):
  """Resets the Android environment, optionally returning to the home screen."""
  for attempt in range(3):
    try:
      app_android_env.reset(go_home=go_home)
      return {"status": "success", "message": f"Environment reset with go_home={go_home}."}
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Reset attempt %d failed: %s. Refreshing controller...", attempt + 1, e)
      if attempt < 2:
        time.sleep(2)
        app_android_env.controller.refresh_env()
      else:
        raise fastapi.HTTPException(status_code=500, detail=str(e)) from e


@app.post("/screenshot")
async def get_screenshot(wait_to_stabilize: bool, app_android_env: AndroidEnv):
  """Captures and returns the current screenshot of the Android environment."""
  if wait_to_stabilize:
    # Pixel-level stabilization: waits for two consecutive identical
    # captures WITHOUT touching the (fragile) a11y/uiautomator chain, so a
    # broken UI-tree pipeline can no longer fail a stabilized screenshot.
    # Falls back to the full state only if the pixel path itself fails.
    try:
      pixels = app_android_env.controller.get_stable_pixels()
    except Exception:
      logger.exception("get_stable_pixels failed; falling back to get_state")
      pixels = app_android_env.get_state(wait_to_stabilize=True).pixels
  else:
    # Pixels-only fast path: never touches the (fragile) a11y/uiautomator
    # chain, so a broken UI-tree pipeline cannot fail a screenshot. Falls
    # back to the full state only if both independent pixel sources fail.
    try:
      pixels = app_android_env.controller.get_pixels()
    except Exception:
      logger.exception("get_pixels failed; falling back to get_state")
      pixels = app_android_env.get_state(wait_to_stabilize=False).pixels
  buf = io.BytesIO()
  Image.fromarray(pixels).save(buf, format='JPEG', quality=85)
  return {"image_b64": base64.b64encode(buf.getvalue()).decode()}


@app.post("/execute_action")
async def execute_action(
    action_dict: dict[str, typing.Any], app_android_env: AndroidEnv
):
  """Executes a given JSON-formatted action in the Android environment."""
  action = json_action.JSONAction(**action_dict)
  app_android_env.execute_action(action)
  return {"status": "success", "message": f"Action {action} executed."}


@suite_router.get("/task_list")
async def suite_task_list(max_index: int, app_suite: AndroidSuite):
  """Returns a list of task keys from the current suite, up to max_index."""
  if max_index > len(app_suite) or max_index < 0:
    return {"task_list": list(app_suite.keys())}
  return {"task_list": list(app_suite.keys())[:max_index]}


@suite_router.get("/task_length")
async def suite_task_length(task_type: str, app_suite: AndroidSuite):
  """Returns the number of tasks for a given task type in the suite."""
  return {"length": len(app_suite[task_type])}


@suite_router.post("/reinitialize")
def reinitialize_suite(
    request: fastapi.Request,
    n_task_combinations: int = 2,  # Default from initial lifespan setup
    seed: int = 42,  # Default from initial lifespan setup
    task_family: str = "android_world",
):
  """Re-initializes the task suite with new parameters."""
  task_registry = request.app.state.task_registry
  try:
    current_aw_registry = task_registry.get_registry(task_family)
  except ValueError as exc:
    raise fastapi.HTTPException(
        status_code=400, detail=f"Invalid task family: {task_family}"
    ) from exc
  new_suite = suite_utils.create_suite(
      task_registry=current_aw_registry,
      n_task_combinations=n_task_combinations,
      seed=seed,
  )
  request.app.state.suite = new_suite
  return {
      "status": "success",
      "message": (
          "Task suite re-initialized with"
          f" n_task_combinations={n_task_combinations}, seed={seed}."
      ),
  }


@task_router.post("/initialize")
async def initialize_task(
    task_type: str,
    task_idx: int,
    app_android_env: AndroidEnv,
    app_suite: AndroidSuite,
):
  """Initializes a specific task in the Android environment.

  On failure the task's `initialized` flag is rolled back so that a client
  retry re-runs the initialization instead of tripping over
  "initialize_task() is already called" forever. (Task subclasses run their
  device setup -- add_contact, send_sms, ... -- AFTER the base class sets
  the flag, so a mid-setup exception used to leave the task in a
  half-initialized limbo that every subsequent call rejected.)
  """
  task = app_suite[task_type][task_idx]
  try:
    task.initialize_task(app_android_env)
  except Exception:
    # Best-effort rollback so the task can be initialized again. tear_down
    # is the sanctioned flag-reset path but also resets device state, which
    # is too heavy to invoke from an error handler; a bare flag reset lets
    # the retry redo setup (subclasses tolerate re-running their steps --
    # e.g. contacts get re-added, and tear_down deletes all contacts
    # anyway).
    try:
      task.initialized = False
      logger.warning(
          "initialize_task(%s[%d]) failed; rolled back initialized flag so"
          " a retry can re-run setup.",
          task_type,
          task_idx,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception("Failed to roll back initialized flag.")
    raise
  return {
      "status": "success",
      "message": f"Task {task_type} {task_idx} initialized.",
  }


@task_router.post("/start_on_home_screen")
async def start_on_home_screen(task_type: str, task_idx: int, app_suite: AndroidSuite):
    start_on_home_screen = app_suite[task_type][task_idx].start_on_home_screen
    return {"start_on_home_screen": start_on_home_screen}


@task_router.post("/complexity")
async def get_task_complexity(task_type: str, task_idx: int, app_suite: AndroidSuite):
    return {"complexity": app_suite[task_type][task_idx].complexity}


@task_router.post("/tear_down")
async def tear_down_task(
    task_type: str,
    task_idx: int,
    app_android_env: AndroidEnv,
    app_suite: AndroidSuite,
):
  """Tears down a specific task in the Android environment."""
  app_suite[task_type][task_idx].tear_down(app_android_env)
  return {
      "status": "success",
      "message": f"Task {task_type} {task_idx} torn down.",
  }


@task_router.get("/score")
async def get_task_score(
    task_type: str,
    task_idx: int,
    app_android_env: AndroidEnv,
    app_suite: AndroidSuite,
):
  """Gets the success status (score) of a specific task."""
  return {
      "score": app_suite[task_type][task_idx].is_successful(app_android_env)
  }


@task_router.get("/goal")
async def get_task_goal(task_type: str, task_idx: int, app_suite: AndroidSuite):
  """Gets the goal description of a specific task."""
  return {"goal": app_suite[task_type][task_idx].goal}


@task_router.get("/template")
async def get_task_template(
    task_type: str, task_idx: int, app_suite: AndroidSuite
):
  """Gets the template or configuration details of a specific task."""
  return {"template": app_suite[task_type][task_idx].template}


@miniwob_router.get("/is_epidode_terminated")
async def is_epidode_terminated(app_android_env: AndroidEnv):
    return {"is_epidode_terminated": get_episode_reward(app_android_env.controller.env) != 0.0}


@app.post("/close")
async def close(app_android_env: AndroidEnv):
  """Closes the Android environment."""
  app_android_env.close()
  return {"status": "success"}


@app.get("/health")
async def health(request: fastapi.Request):
  """Deep health check: env init + QEMU process + adb device state.

  The old check returned 200 as long as the AsyncEnv object existed --
  even with a dead emulator process, which made it useless as a liveness
  probe (see the 2026-08-23 OOM incident where /health stayed 200 while
  every screenshot 500'd). Now verifies, cheaply:
    1. The environment is initialized.
    2. A QEMU emulator process is alive in this container.
    3. adb reports emulator-5554 in 'device' (fully booted) state.
  Returns 503 with a machine-readable reason on any failure.
  """
  env = getattr(request.app.state, "app_android_env", None)
  if not isinstance(env, interface.AsyncEnv):
    return fastapi.responses.JSONResponse(
        status_code=503,
        content={"status": "error", "reason": "env not initialized"},
    )
  # Probe 1: QEMU process alive (pgrep scans /proc; ~milliseconds).
  try:
    qemu_alive = (
        subprocess.run(
            ["pgrep", "-f", "qemu-system-x86_64-headless"],
            capture_output=True,
            timeout=5,
        ).returncode
        == 0
    )
  except Exception:  # pylint: disable=broad-exception-caught
    qemu_alive = False
  if not qemu_alive:
    return fastapi.responses.JSONResponse(
        status_code=503,
        content={"status": "error", "reason": "qemu process not found"},
    )
  # Probe 2: emulator visible to adb in 'device' state.
  adb_path = getattr(request.app.state, "adb_path", None) or (
      "/opt/android/platform-tools/adb"
  )
  try:
    devices_out = subprocess.run(
        [adb_path, "devices"],
        capture_output=True,
        timeout=5,
    ).stdout.decode("utf-8", errors="replace")
    device_ok = "emulator-5554\tdevice" in devices_out
  except Exception:  # pylint: disable=broad-exception-caught
    device_ok = False
  if not device_ok:
    return fastapi.responses.JSONResponse(
        status_code=503,
        content={"status": "error", "reason": "emulator-5554 not in device state"},
    )
  return {"status": "success"}


@adb_router.post("/start_activity")
async def start_activity(
    activity: str,
    app_android_env: AndroidEnv,
    extra_args: list[str] = fastapi.Query(default=[]),
    timeout_sec: float = 10,
):
  """Launches the given activity."""
  response = adb_utils.start_activity(activity, extra_args, app_android_env.controller, timeout_sec)
  return {"status": response.status, "output": response.generic.output.decode()}


@adb_router.get("/current_activity")
async def get_current_activity(app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Returns the full activity name currently opened."""
  activity, _ = adb_utils.get_current_activity(app_android_env.controller, timeout_sec)
  return {"activity": activity}


@adb_router.post("/tap")
async def tap_screen(x: int, y: int, app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Taps the screen at (x, y)."""
  adb_utils.tap_screen(x, y, app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/double_tap")
async def double_tap(x: int, y: int, app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Double taps the screen at (x, y)."""
  adb_utils.double_tap(x, y, app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/long_press")
async def long_press(x: int, y: int, app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Long presses the screen at (x, y)."""
  adb_utils.long_press(x, y, app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/press_home")
async def press_home_button(app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Presses the HOME button."""
  adb_utils.press_home_button(app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/press_back")
async def press_back_button(app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Presses the BACK button."""
  adb_utils.press_back_button(app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/press_enter")
async def press_enter_button(app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Presses the ENTER button."""
  adb_utils.press_enter_button(app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/press_key")
async def press_keyboard_generic(keycode: str, app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Presses any keyboard key by keycode."""
  adb_utils.press_keyboard_generic(keycode, app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/type_text")
async def type_text(text: str, app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Types the specified text string."""
  adb_utils.type_text(text, app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/generic_request")
async def issue_generic_request(
    args: list[str] | str, app_android_env: AndroidEnv, timeout_sec: float = 10
):
  """Issues a generic adb command."""
  response = adb_utils.issue_generic_request(args, app_android_env.controller, timeout_sec)
  return {"status": response.status, "output": response.generic.output.decode()}


@adb_router.get("/adb_activity")
async def get_adb_activity(app_name: str):
  """Gets the ADB activity for a given app name."""
  return {"activity": adb_utils.get_adb_activity(app_name)}


@adb_router.get("/all_packages")
async def get_all_package_names(app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Returns all installed package names."""
  return {"packages": adb_utils.get_all_package_names(app_android_env.controller, timeout_sec)}


@adb_router.get("/all_apps")
async def get_all_apps(app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Returns all installed app names."""
  return {"apps": adb_utils.get_all_apps(app_android_env.controller, timeout_sec)}


@adb_router.post("/launch_app")
async def launch_app(app_name: str, app_android_env: AndroidEnv):
  """Launches an app by name."""
  result = adb_utils.launch_app(app_name, app_android_env.controller)
  return {"launched": result}


@adb_router.get("/extract_package_name")
async def extract_package_name(activity: str):
  """Extracts the package name from an activity string."""
  return {"package_name": adb_utils.extract_package_name(activity)}


@adb_router.post("/close_recents")
async def close_recents(app_android_env: AndroidEnv):
  """Closes all recent apps."""
  adb_utils.close_recents(app_android_env.controller)
  return {"status": "success"}


@adb_router.post("/close_app")
async def close_app(app_name: str, app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Closes an app by name."""
  result = adb_utils.close_app(app_name, app_android_env.controller, timeout_sec)
  return {"closed": result}


@adb_router.get("/generate_swipe_command")
async def generate_swipe_command(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration_ms: Optional[int] = None,
):
  """Generates a swipe adb command argument list."""
  return {"command": adb_utils.generate_swipe_command(start_x, start_y, end_x, end_y, duration_ms)}


@adb_router.get("/generate_drag_and_drop_command")
async def generate_drag_and_drop_command(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration_ms: Optional[int] = None,
):
  """Generates a drag and drop adb command argument list."""
  return {"command": adb_utils.generate_drag_and_drop_command(start_x, start_y, end_x, end_y, duration_ms)}


@adb_router.post("/send_intent")
async def send_android_intent(body: SendIntentRequest, app_android_env: AndroidEnv):
  """Sends an Android intent."""
  response = adb_utils.send_android_intent(
      body.command, body.action, app_android_env.controller,
      body.data_uri, body.mime_type, body.extras, body.timeout_sec,
  )
  return {"status": response.status, "output": response.generic.output.decode()}


@adb_router.get("/api_level")
async def get_api_level(app_android_env: AndroidEnv):
  """Gets the API level of the device."""
  return {"api_level": adb_utils.get_api_level(app_android_env.controller)}


@adb_router.post("/toggle_wifi")
async def toggle_wifi(on_or_off: str, app_android_env: AndroidEnv):
  """Toggles wifi on or off."""
  adb_utils.toggle_wifi(app_android_env.controller, on_or_off)  # type: ignore[arg-type]
  return {"status": "success"}


@adb_router.post("/toggle_bluetooth")
async def toggle_bluetooth(on_or_off: str, app_android_env: AndroidEnv):
  """Toggles Bluetooth on or off."""
  adb_utils.toggle_bluetooth(app_android_env.controller, on_or_off)  # type: ignore[arg-type]
  return {"status": "success"}


@adb_router.post("/set_brightness")
async def set_brightness(max_or_min: str, app_android_env: AndroidEnv):
  """Sets screen brightness to max or min."""
  adb_utils.set_brightness(max_or_min, app_android_env.controller)
  return {"status": "success"}


@adb_router.post("/clear_app_data")
async def clear_app_data(package_name: str, app_android_env: AndroidEnv):
  """Clears all data for a given package."""
  adb_utils.clear_app_data(package_name, app_android_env.controller)
  return {"status": "success"}


@adb_router.post("/toggle_airplane_mode")
async def toggle_airplane_mode(on_or_off: str, app_android_env: AndroidEnv):
  """Toggles airplane mode on or off."""
  adb_utils.toggle_airplane_mode(on_or_off, app_android_env.controller)  # type: ignore[arg-type]
  return {"status": "success"}


@adb_router.post("/install_apk")
async def install_apk(apk_location: str, app_android_env: AndroidEnv):
  """Installs an APK."""
  adb_utils.install_apk(apk_location, app_android_env.controller)
  return {"status": "success"}


@adb_router.get("/airplane_mode")
async def check_airplane_mode(app_android_env: AndroidEnv):
  """Checks if airplane mode is enabled."""
  return {"enabled": adb_utils.check_airplane_mode(app_android_env.controller)}


@adb_router.post("/extract_broadcast_data")
async def extract_broadcast_data(raw_output: str):
  """Extracts data from an adb broadcast command output."""
  return {"data": adb_utils.extract_broadcast_data(raw_output)}


@adb_router.get("/clipboard")
async def get_clipboard_contents(app_android_env: AndroidEnv):
  """Gets the clipboard content."""
  return {"content": adb_utils.get_clipboard_contents(app_android_env.controller)}


@adb_router.post("/change_orientation")
async def change_orientation(orientation: str, app_android_env: AndroidEnv):
  """Changes the screen orientation."""
  adb_utils.change_orientation(orientation, app_android_env.controller)
  return {"status": "success"}


@adb_router.post("/set_clipboard")
async def set_clipboard_contents(content: str, app_android_env: AndroidEnv):
  """Sets the clipboard content."""
  adb_utils.set_clipboard_contents(content, app_android_env.controller)
  return {"status": "success"}


@adb_router.post("/grant_permissions")
async def grant_permissions(
    activity_name: str, permission: str, app_android_env: AndroidEnv
):
  """Grants a permission to an activity."""
  adb_utils.grant_permissions(activity_name, permission, app_android_env.controller)
  return {"status": "success"}


@adb_router.post("/execute_sql")
async def execute_sql_command(
    db_path: str, sql_command: str, app_android_env: AndroidEnv
):
  """Executes an SQL command on a SQLite database via ADB."""
  response = adb_utils.execute_sql_command(db_path, sql_command, app_android_env.controller)
  return {"status": response.status, "output": response.generic.output.decode()}


@adb_router.get("/call_state")
async def get_call_state(app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Gets the current call state."""
  return {"state": adb_utils.get_call_state(app_android_env.controller, timeout_sec)}


@adb_router.post("/call_emulator")
async def call_emulator(
    phone_number: str, app_android_env: AndroidEnv, timeout_sec: float = 10
):
  """Simulates an incoming call in the emulator."""
  adb_utils.call_emulator(app_android_env.controller, phone_number, timeout_sec)
  return {"status": "success"}


@adb_router.post("/end_call")
async def end_call_if_active(app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Ends the phone call if active."""
  adb_utils.end_call_if_active(app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/clear_call_log")
async def clear_android_emulator_call_log(
    app_android_env: AndroidEnv, timeout_sec: float = 10
):
  """Clears the call log."""
  adb_utils.clear_android_emulator_call_log(app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/call_phone")
async def call_phone_number(
    phone_number: str, app_android_env: AndroidEnv, timeout_sec: float = 10
):
  """Initiates a phone call."""
  adb_utils.call_phone_number(app_android_env.controller, phone_number, timeout_sec)
  return {"status": "success"}


@adb_router.post("/text_emulator")
async def text_emulator(
    phone_number: str, message: str, app_android_env: AndroidEnv, timeout_sec: float = 10
):
  """Simulates an incoming SMS in the emulator."""
  adb_utils.text_emulator(app_android_env.controller, phone_number, message, timeout_sec)
  return {"status": "success"}


@adb_router.post("/set_default_app")
async def set_default_app(
    setting_key: str, package_name: str, app_android_env: AndroidEnv, timeout_sec: float = 10
):
  """Sets the default app for a given setting key."""
  adb_utils.set_default_app(setting_key, package_name, app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/disable_headsup")
async def disable_headsup_notifications(
    app_android_env: AndroidEnv, timeout_sec: float = 10
):
  """Disables heads-up notifications."""
  adb_utils.disable_headsup_notifications(app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/enable_headsup")
async def enable_headsup_notifications(
    app_android_env: AndroidEnv, timeout_sec: float = 10
):
  """Enables heads-up notifications."""
  adb_utils.enable_headsup_notifications(app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.post("/put_settings")
async def put_settings(
    namespace: int, key: str, value: str, app_android_env: AndroidEnv
):
  """Changes a system setting via ADB."""
  adb_utils.put_settings(namespace, key, value, app_android_env.controller)
  return {"status": "success"}


@adb_router.get("/all_settings")
async def get_all_settings(app_android_env: AndroidEnv):
  """Gets all system settings."""
  return {"settings": adb_utils.get_all_settings(app_android_env.controller)}


@adb_router.post("/delete_contacts")
async def delete_contacts(app_android_env: AndroidEnv, timeout_sec: float = 10):
  """Deletes all contacts."""
  adb_utils.delete_contacts(app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.get("/screen_size")
async def get_screen_size(app_android_env: AndroidEnv):
  """Gets the physical screen size in pixels."""
  width, height = adb_utils.get_screen_size(app_android_env.controller)
  return {"width": width, "height": height}


@adb_router.get("/logical_screen_size")
async def get_logical_screen_size(app_android_env: AndroidEnv):
  """Gets the logical screen size."""
  width, height = adb_utils.get_logical_screen_size(app_android_env.controller)
  return {"width": width, "height": height}


@adb_router.get("/physical_frame_boundary")
async def get_physical_frame_boundary(app_android_env: AndroidEnv):
  """Gets the physical frame boundary."""
  x1, y1, x2, y2 = adb_utils.get_physical_frame_boundary(app_android_env.controller)
  return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


@adb_router.get("/orientation")
async def get_orientation(app_android_env: AndroidEnv):
  """Gets the current screen orientation."""
  return {"orientation": adb_utils.get_orientation(app_android_env.controller)}


@adb_router.post("/set_screen_size")
async def set_screen_size(width: int, height: int, app_android_env: AndroidEnv):
  """Sets the logical screen size."""
  adb_utils.set_screen_size(width, height, app_android_env.controller)
  return {"status": "success"}


_RETRY_ALLOWLIST = frozenset({
    "get_current_activity",
    "get_orientation",
    "get_screen_size",
    "get_logical_screen_size",
    "get_api_level",
    "get_all_package_names",
    "get_all_apps",
    "get_clipboard_contents",
    "check_airplane_mode",
    "get_call_state",
    "uiautomator_dump",
})


@adb_router.post("/retry")
async def retry(n: int, func_name: str, app_android_env: AndroidEnv):
  """Retries an adb_utils function up to n times on AdbControllerError."""
  if func_name not in _RETRY_ALLOWLIST:
    raise fastapi.HTTPException(status_code=400, detail=f"Function not allowed: {func_name}")
  func = getattr(adb_utils, func_name)
  retrying = adb_utils.retry(n)(func)
  result = retrying(app_android_env.controller)
  return {"status": "success", "result": str(result)}


@adb_router.post("/set_root")
async def set_root_if_needed(app_android_env: AndroidEnv, timeout_sec: Optional[float] = None):
  """Sets ADB to root if not already."""
  adb_utils.set_root_if_needed(app_android_env.controller, timeout_sec)
  return {"status": "success"}


@adb_router.get("/uiautomator_dump")
async def uiautomator_dump(app_android_env: AndroidEnv, timeout_sec: float = 30):
  """Returns the UI hierarchy via uiautomator dump."""
  return {"ui_hierarchy": adb_utils.uiautomator_dump(app_android_env.controller, timeout_sec)}


task_router.include_router(miniwob_router)
app.include_router(suite_router)
app.include_router(task_router)
app.include_router(adb_router)

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=5000)
