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

"""Run eval suite against an Android environment running inside Docker.

This is the Docker counterpart of run.py. Instead of loading a local emulator
and running the task suite in-process, it talks over HTTP to the environment
server in the container (see server/android_server.py); the emulator, the
environment, and the task objects all live server-side. The task lifecycle
(initialize / goal / complexity / start_on_home_screen / score / tear_down)
is driven entirely through the client's /task/* and /suite/* endpoints, while
the agent reads state and sends actions through the same client.

Prerequisites (see README's "Docker Support" section):

  # Build the image once:
  docker build -t android_world:latest .
  # Then start a container on host port 5000 (takes 5-10 minutes to boot):
  ./scripts/run_container.sh 5000

Example:

  python run_on_docker.py --agent_name client_t3a \
      --suite_family android_world --tasks ClockStopWatchRunning

To run only part of a family, pass --task_index_range. Positions are 1-based
and count over the whole suite (not over --tasks), so the 116-task
android_world family splits cleanly into two runs:

  python run_on_docker.py --agent_name client_t3a \
      --suite_family android_world --task_index_range 59-116

Notes:
  * The container is NOT shut down at the end of the run -- see
    `_main`'s closing comment for why /close is deliberately not called.
  * Cross-path checkpoint resume (a run started by run.py resumed here, or
    vice versa) works for the `android_world` family, whose registry keys
    equal the task class names. For the MiniWoB families the registry keys
    are the HTML task names (e.g. 'book-flight') while run.py checkpoints
    under the class name (e.g. 'MiniWobBookFlight'), so those two paths do
    not share checkpoints.
"""

from collections.abc import Sequence
import datetime
import hashlib
import json
import os
import re
import time
import traceback
from typing import Any

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
from android_world import checkpointer as checkpointer_lib
from android_world import constants
from android_world import episode_runner
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import generic_v2
from android_world.agents import infer
from android_world.agents import mobile_jev
from android_world.agents import mobile_jev_v2
from android_world.agents import t3a
from android_world.env import interface
from android_world.env import representation_utils

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


_SERVER_URL = flags.DEFINE_string(
    'server_url',
    'http://localhost:5000',
    'Base URL of the Android environment server running in Docker (see'
    ' scripts/run_container.sh).',
)

_BASE_URL = flags.DEFINE_string(
    'base_url',
    'https://api.openai.com',
    'Base URL for the OpenAI API. Set to a custom endpoint if using a proxy or'
    ' alternative service.',
)

_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    'gpt-4-turbo-2024-04-09',
    'The model name to use for the OpenAI API. This should be a valid model'
    ' name supported by the OpenAI API.',
)

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    'Suite family to run. See registry.py for more information.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 30, 'Random seed for task randomness.'
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)

_TASK_INDEX_RANGE = flags.DEFINE_string(
    'task_index_range',
    None,
    'Range of tasks to run, in the format "START-END" (1-based, inclusive).'
    ' For example, "1-58" runs the 1st through 58th task in the suite.'
    ' If None, all tasks are run.',
)

_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from. If the'
    ' directory contains existing checkpoint files, evaluation will resume from'
    ' the latest checkpoint. If the directory is empty or does not exist, a new'
    ' directory will be created.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    os.path.expanduser('~/android_world/runs'),
    'The path to save results to if not resuming from a checkpoint is not'
    ' provided.',
)

_RAW_DUMPS = flags.DEFINE_bool(
    'raw_dumps',
    True,
    'Write each episode\'s images and text data as PNG + JSON under'
    ' <checkpoint_dir>/raw_dumps/<task>_<instance>/ so it can be inspected'
    ' without unpickling. The .pkl.gz episode logs are always written.',
)

# Agent specific.
_AGENT_NAME = flags.DEFINE_string(
    'agent_name', 'client_t3a', help='Agent name.'
)

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed when running multiple task combinations'
    ' (n_task_combinations > 1).',
)

_HEALTH_TIMEOUT_SEC = flags.DEFINE_integer(
    'health_timeout_sec',
    900,
    'How long to wait for the environment server to become healthy. A fresh'
    ' container needs 5-10 minutes to boot the emulator.',
)
_HEALTH_POLL_INTERVAL_SEC = 5.0


# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

# Additional guidelines for the MiniWob tasks.
_MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        'This task is running in a mock app, you must stay in this app and'
        ' DO NOT use the `navigate_home` action.'
    ),
]

# Fields loaded from an existing checkpoint when deciding what to resume.
# Keep in sync with suite_utils._run_task_suite.
_METADATA_FIELDS = [
    constants.EpisodeConstants.GOAL,
    constants.EpisodeConstants.TASK_TEMPLATE,
    constants.EpisodeConstants.INSTANCE_ID,
    constants.EpisodeConstants.IS_SUCCESSFUL,
    constants.EpisodeConstants.EPISODE_LENGTH,
    constants.EpisodeConstants.RUN_TIME,
    constants.EpisodeConstants.EXCEPTION_INFO,
    constants.EpisodeConstants.AUX_DATA,
]


def _get_agent(
    client: interface.AndroidEnvClient,
    family: str | None = None,
) -> base_agent.ClientInteractingAgent:
  """Gets agent that talks to the Docker environment server.

  Args:
    client: The environment client.
    family: Suite family, used to attach MiniWoB-specific guidelines.

  Returns:
    A client-backed agent.

  Raises:
    ValueError: If the agent name is unknown or not supported in Docker mode.
  """
  print('Initializing agent...')
  agent = None
  if _AGENT_NAME.value == 'client_t3a':
    agent = t3a.ClientT3A(
        client, infer.OpenAIWrapper(base_url=_BASE_URL.value, model_name=_MODEL_NAME.value)
    )
  elif _AGENT_NAME.value == 'client_generic':
    # The baseline GenericAgentV2 port: the upstream MobileGym prompt, parsing
    # and history, without the R2-SOL gates. Use client_generic_r2sol for the
    # evolved candidate so the two can be compared directly.
    agent = generic_v2.ClientGeneric(
        client, base_url=_BASE_URL.value, model_name=_MODEL_NAME.value
    )
  elif _AGENT_NAME.value == 'client_generic_r2sol':
    # This agent calls the model itself so it can honor the upstream sampling
    # parameters, which OpenAIWrapper cannot express.
    agent = generic_v2.ClientGenericR2SOL(
        client, base_url=_BASE_URL.value, model_name=_MODEL_NAME.value
    )
  elif _AGENT_NAME.value == 'client_mobile_jev':
    # Jev (TypeSafe) makes every decision; the LlmWrapper slot is unused.
    # TYPESAFE_API_KEY / TYPESAFE_MODEL come from the environment; --base_url
    # and --model_name are irrelevant for this agent.
    agent = mobile_jev.ClientMobileJev(
        client, llm=None, jev=infer.TypeSafeJevWrapper()
    )
  elif _AGENT_NAME.value == 'client_mobile_jev_v2':
    # The ac-jev-v2 checkpoint decides from the screenshots, so this agent also
    # needs a multimodal LLM -- the one job the decision model was not trained
    # for is naming the text a TYPE_TEXT types. TYPESAFE_API_KEY and
    # TYPESAFE_MODEL come from the environment, as does the vision endpoint
    # (TYPESAFE_MM_ENDPOINT) and the completion cut (JEV_DONE_THRESHOLD).
    agent = mobile_jev_v2.ClientMobileJevV2(
        client,
        llm=infer.OpenAIWrapper(
            base_url=_BASE_URL.value, model_name=_MODEL_NAME.value
        ),
        jev=infer.TypeSafeJevWrapper(),
        done_threshold=float(
            os.environ.get('JEV_DONE_THRESHOLD', '').strip()
            or mobile_jev_v2.DONE_THRESHOLD
        ),
    )
  else:
    raise ValueError(
        f'Unknown agent for Docker mode: {_AGENT_NAME.value}. Client-backed'
        ' agents currently available: client_t3a, client_generic,'
        ' client_generic_r2sol, client_mobile_jev, client_mobile_jev_v2. (The'
        ' env-backed agents'
        ' from run.py -- human_agent, random_agent, m3a_*, t3a_*, seeact --'
        ' require a local emulator and are not usable against the Docker'
        ' server.)'
    )

  if (
      agent.name in ['M3A', 'T3A', 'SeeAct', 'ClientT3A']
      and family
      and family.startswith('miniwob')
      and hasattr(agent, 'set_task_guidelines')
  ):
    agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
  agent.name = _AGENT_NAME.value

  return agent


def _wait_for_healthy_server(
    client: interface.AndroidEnvClient, timeout_sec: float
) -> None:
  """Blocks until the environment server is healthy or the timeout expires.

  The server's /health is a deep probe (env initialized + QEMU process alive
  + emulator visible to adb), so it is a reliable readiness signal.

  Args:
    client: The environment client.
    timeout_sec: Maximum total time to wait.

  Raises:
    RuntimeError: If the server did not become healthy in time.
  """
  start = time.time()
  attempt = 0
  while True:
    attempt += 1
    if client.health():
      print(f'Environment server is healthy (took {time.time() - start:.0f}s).')
      return
    elapsed = time.time() - start
    if elapsed >= timeout_sec:
      raise RuntimeError(
          f'Environment server at {client.base_url} was not healthy after'
          f' {elapsed:.0f}s (attempt {attempt}). Check the container logs:'
          f' docker logs aw_5000. A fresh container normally needs 5-10'
          ' minutes to boot the emulator.'
      )
    print(
        f'Environment not healthy yet (attempt {attempt}, {elapsed:.0f}s'
        f' elapsed); retrying in {_HEALTH_POLL_INTERVAL_SEC:.0f}s...'
    )
    time.sleep(_HEALTH_POLL_INTERVAL_SEC)


def _parse_task_index_range(spec: str) -> tuple[int, int]:
  """Parses a "START-END" task range into 1-based inclusive bounds.

  Args:
    spec: The range, e.g. "1-58".

  Returns:
    The (start, end) bounds, both 1-based inclusive.

  Raises:
    ValueError: If `spec` is not "START-END" with 1 <= START <= END.
  """
  match = re.fullmatch(r'(\d+)\s*-\s*(\d+)', spec.strip())
  if match is None:
    raise ValueError(
        f'Invalid --task_index_range {spec!r}: expected "START-END"'
        ' (1-based, inclusive), e.g. "1-58".'
    )
  start, end = int(match.group(1)), int(match.group(2))
  if start < 1:
    raise ValueError(
        f'Invalid --task_index_range {spec!r}: positions are 1-based, so'
        ' START must be at least 1.'
    )
  if end < start:
    raise ValueError(
        f'Invalid --task_index_range {spec!r}: START ({start}) must not'
        f' exceed END ({end}).'
    )
  return start, end


def _slice_task_index_range(
    task_list: list[str], index_range: str | None
) -> list[str]:
  """Keeps the suite positions covered by `index_range`.

  Positions count over the whole suite, independent of any `--tasks` filter,
  so "1-58" always means the same tasks. Out-of-bounds ranges raise instead of
  silently selecting nothing: "59-116" on a 58-task suite is a mistake worth
  hearing about.

  Args:
    task_list: The suite's task keys, in suite order.
    index_range: The "START-END" range, or None to keep everything.

  Returns:
    The selected task keys, in suite order.

  Raises:
    ValueError: If `index_range` is malformed or exceeds the suite size.
  """
  if index_range is None:
    return list(task_list)
  start, end = _parse_task_index_range(index_range)
  if end > len(task_list):
    raise ValueError(
        f'--task_index_range {index_range!r} is out of bounds: the suite has'
        f' {len(task_list)} tasks, so END must be at most {len(task_list)}.'
    )
  return task_list[start - 1 : end]


def _select_tasks(
    client: interface.AndroidEnvClient,
    tasks: list[str] | None,
    index_range: str | None = None,
) -> list[str]:
  """Returns the suite's task keys, filtered to `tasks` and `index_range`.

  Validation is done against the server's suite (not a local registry), since
  the server owns the task objects. When both filters are given, `index_range`
  still counts over the whole suite and the two are intersected -- so
  combining them keeps only the tasks that satisfy both.

  Args:
    client: The environment client.
    tasks: Task keys to keep, or None to keep all.
    index_range: "START-END" suite positions (1-based, inclusive) to keep, or
      None to keep all.

  Returns:
    The task keys to run, in suite order.

  Raises:
    ValueError: If a requested task is not in the suite, `index_range` is
      malformed or out of bounds, or the two filters select nothing.
  """
  task_list = client.get_suite_task_list(max_index=-1)
  selected = _slice_task_index_range(task_list, index_range)
  if tasks is None:
    return selected
  for name in tasks:
    if name not in task_list:
      raise ValueError(
          f'Task {name} not found in the suite.'
          + suite_utils._suggest_keyword(name, task_list)  # pylint: disable=protected-access
      )
  selected = [name for name in selected if name in set(tasks)]
  if not selected and index_range is not None:
    raise ValueError(
        f'--tasks and --task_index_range {index_range!r} select no tasks in'
        ' common: positions count over the whole suite, not over --tasks.'
    )
  return selected


def _instance_seed(
    seed: int | None, task_type: str, task_idx: int, use_identical_params: bool
) -> int | None:
  """Recomputes the seed suite_utils gave an instance of a server-side suite.

  `suite_utils.create_suite` seeds each instance with a sha256 over
  f'{seed}_{name}_{i}' (or over instance 0 when use_identical_params is set).
  That helper is a closure inside create_suite, so the formula is duplicated
  here; it must stay in sync or the recorded seed will not match the params
  the server actually used.

  Args:
    seed: The suite's random seed.
    task_type: The task key.
    task_idx: The instance index within the task.
    use_identical_params: Whether the suite shares params across instances.

  Returns:
    The instance's seed, or None if the suite was created without a seed.
  """
  if seed is None:
    return None
  index = 0 if use_identical_params else task_idx
  unique_seed_str = f'{seed}_{task_type}_{index}'
  return int(hashlib.sha256(unique_seed_str.encode()).hexdigest(), 16) % (
      2**32
  )


# The frames an agent may capture per step, in the order the dumps prefer them:
# the before screenshot is the screen the agent observed when it made its
# decision, so it is the most useful frame for replay and debugging. Both hold
# pixel arrays, so neither is ever copied into the JSON.
_STEP_FRAME_KEYS = ('before_screenshot', 'after_screenshot')


def _write_json(path: str, payload: Any) -> None:
  """Writes one JSON file, stringifying values JSON cannot represent."""
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _step_frame(step_data: dict[str, Any], index: int) -> np.ndarray | None:
  """Returns the frame captured for one step, or None if it captured none.

  Args:
    step_data: The transposed episode step data.
    index: The step's position in the transposed lists.

  Returns:
    The step's frame, if any, preferring the before screenshot.
  """
  for key in _STEP_FRAME_KEYS:
    frames = step_data.get(key) or []
    pixels = frames[index] if index < len(frames) else None
    if isinstance(pixels, np.ndarray) and pixels.ndim == 3:
      return pixels
  return None


def _write_step_image(
    pixels: np.ndarray, dump_dir: str, filename: str
) -> None:
  """Saves one step's frame as a PNG."""
  Image.fromarray(pixels).save(os.path.join(dump_dir, filename), format='PNG')


def _dump_raw_episode(
    checkpoint_dir: str, instance_name: str, episode: dict[str, Any]
) -> str:
  """Writes an episode's images and text data next to its .pkl.gz log.

  Layout: <checkpoint_dir>/raw_dumps/<instance_name>/ containing an
  episode.json summary, and for every step a step_NNN.json with the model
  request/response, the element lists, the agent diagnostics and the timings,
  plus a matching step_NNN.png holding the step's frame.

  Args:
    checkpoint_dir: The run's checkpoint directory.
    instance_name: The task instance key, e.g. 'ClockStopWatchRunning_0'.
    episode: The episode record that was passed to the checkpointer.

  Returns:
    The directory the dumps were written to.
  """
  episode_constants = constants.EpisodeConstants
  step_data = episode.get(episode_constants.EPISODE_DATA) or {}
  step_numbers = step_data.get(constants.STEP_NUMBER) or []
  dump_dir = os.path.join(checkpoint_dir, 'raw_dumps', instance_name)
  os.makedirs(dump_dir, exist_ok=True)
  _write_json(
      os.path.join(dump_dir, 'episode.json'),
      {
          'instance': instance_name,
          'task': episode.get(episode_constants.TASK_TEMPLATE),
          'agent': episode.get(episode_constants.AGENT_NAME),
          'goal': episode.get(episode_constants.GOAL),
          'is_successful': episode.get(episode_constants.IS_SUCCESSFUL),
          'run_time': episode.get(episode_constants.RUN_TIME),
          'episode_length': episode.get(episode_constants.EPISODE_LENGTH),
          'seed': episode.get(episode_constants.SEED),
          'exception_info': episode.get(episode_constants.EXCEPTION_INFO),
      },
  )
  # A step that captured no frame of its own -- the agent decided it was done
  # without touching the device -- inherits the previous capture, which is
  # still the current screen, so every step_NNN.json pairs with a step_NNN.png.
  frame: np.ndarray | None = None
  frame_step: int | None = None
  for index, step_number in enumerate(step_numbers):
    step = step_number + 1
    prefix = f'step_{step:03d}'
    captured = _step_frame(step_data, index)
    if captured is not None:
      frame, frame_step = captured, step
    if frame is not None:
      _write_step_image(frame, dump_dir, f'{prefix}.png')
    payload: dict[str, Any] = {
        'instance': instance_name,
        'step': step,
        'goal': episode.get(episode_constants.GOAL),
        'agent': episode.get(episode_constants.AGENT_NAME),
        'screenshot': f'{prefix}.png' if frame is not None else None,
    }
    if frame is not None and frame_step != step:
      payload['screenshot_from_step'] = frame_step
    for key, values in step_data.items():
      if key in _STEP_FRAME_KEYS or key == constants.STEP_NUMBER:
        continue
      value = (
          values[index]
          if isinstance(values, list) and index < len(values)
          else None
      )
      if (
          isinstance(value, list)
          and value
          and isinstance(value[0], representation_utils.UIElement)
      ):
        value = [representation_utils.ui_element_to_dict(e) for e in value]
      payload[key] = value
    _write_json(os.path.join(dump_dir, f'{prefix}.json'), payload)
  return dump_dir


def _build_episode(
    task_type: str,
    goal: str,
    episode: episode_runner.EpisodeResult,
    task_score: float,
    run_time: float,
    seed: int | None,
) -> dict[str, Any]:
  """Builds the episode record for the checkpointer.

  Mirrors suite_utils._run_task's success record (including its
  agent_successful rule: a task only counts if the agent indicated done).

  Args:
    task_type: The task key.
    goal: The task goal.
    episode: The result of running the agent on the task.
    task_score: Score reported by the server for the task.
    run_time: Wall-clock time of the episode.
    seed: The task instance's random seed.

  Returns:
    The episode record.
  """
  agent_successful = task_score if episode.done else 0.0
  return {
      constants.EpisodeConstants.GOAL: goal,
      constants.EpisodeConstants.TASK_TEMPLATE: task_type,
      constants.EpisodeConstants.EPISODE_DATA: episode.step_data,
      constants.EpisodeConstants.IS_SUCCESSFUL: agent_successful,
      constants.EpisodeConstants.RUN_TIME: run_time,
      constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
      constants.EpisodeConstants.EPISODE_LENGTH: len(
          episode.step_data[constants.STEP_NUMBER]
      ),
      constants.EpisodeConstants.AUX_DATA: episode.aux_data,
      # The runner has no task object, so it cannot read a per-task screen
      # config; the container emulator is a Pixel 6 (1080x2400, portrait).
      constants.EpisodeConstants.SCREEN_CONFIG: {
          'width': 1080,
          'height': 2400,
          'orientation': 'portrait',
          'config_name': 'default',
      },
      constants.EpisodeConstants.EXCEPTION_INFO: None,
      constants.EpisodeConstants.SEED: seed,
  }


def _run_instance(
    client: interface.AndroidEnvClient,
    agent: base_agent.ClientInteractingAgent,
    task_type: str,
    task_idx: int,
    seed: int | None,
    is_miniwob: bool,
) -> dict[str, Any]:
  """Runs one task instance, mirroring suite_utils._run_task.

  Args:
    client: The environment client.
    agent: The agent to run.
    task_type: The task key.
    task_idx: The instance index within the task.
    seed: The task instance's random seed.
    is_miniwob: Whether the suite family is a MiniWoB family, which needs the
      episode-termination check each step. (suite_utils.run keys this off
      `task.name.startswith('miniwob')`; in Docker mode only the registry key
      is available -- e.g. 'book-flight' -- so the family is used instead.
      MiniWoB families contain only MiniWoB tasks, so the two agree.)

  Returns:
    The episode record (successful or failed).
  """
  start = time.time()
  goal = ''
  try:
    # initialize_task must precede get_task_goal: MiniWoB goals are only
    # available after the task initialized the episode on the device.
    client.initialize_task(task_type, task_idx)
    goal = client.get_task_goal(task_type, task_idx)
    complexity = client.get_task_complexity(task_type, task_idx)
    if complexity is None:
      raise ValueError('Task complexity must be provided.')
    start_on_home_screen = client.start_on_home_screen(task_type, task_idx)

    suite_utils._log_and_print(  # pylint: disable=protected-access
        'Running task %s with goal "%s"', task_type, goal
    )
    episode = episode_runner.run_episode(
        goal=goal,
        agent=agent,
        max_n_steps=int(10 * complexity),
        start_on_home_screen=start_on_home_screen,
        termination_fn=(
            (lambda _env: client.is_miniwob_episode_terminated())
            if is_miniwob
            else None
        ),
    )
    # Scored inside the try: like suite_utils._run_task's is_successful call,
    # a failure here is an episode failure, not a silent 0.0.
    task_score = client.get_task_score(task_type, task_idx)
    episode_record = _build_episode(
        task_type, goal, episode, task_score, time.time() - start, seed
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    suite_utils._log_and_print(  # pylint: disable=protected-access
        '%s\nSKIPPING %s.', '~' * 80, task_type
    )
    logging.exception(
        'Logging exception and skipping task. Will keep running. Task: %s: %s',
        task_type,
        e,
    )
    traceback.print_exc()
    return suite_utils._create_failed_result(  # pylint: disable=protected-access
        task_type, goal, traceback.format_exc(), time.time() - start
    )
  else:
    suite_utils._log_and_print(  # pylint: disable=protected-access
        '%s; %s',
        'Task Successful ✅'
        if episode_record[constants.EpisodeConstants.IS_SUCCESSFUL] > 0.5
        else 'Task Failed ❌',
        f' {goal}',
    )
    # Mirrors suite_utils._run_task: tear_down runs after the record is
    # assembled, so a tear_down failure propagates instead of losing the
    # episode.
    client.tear_down_task(task_type, task_idx)
    return episode_record


def _main() -> None:
  """Runs eval suite against the Docker environment server."""
  n_task_combinations = _N_TASK_COMBINATIONS.value
  use_identical_params = _FIXED_TASK_SEED.value

  # Fail fast on a malformed range: _select_tasks validates it again once the
  # suite is known, but that happens after the health wait, and a typo should
  # not cost a 5-10 minute container boot.
  if _TASK_INDEX_RANGE.value is not None:
    _parse_task_index_range(_TASK_INDEX_RANGE.value)

  # A v2 run cannot make a single decision without its vision endpoint; fail
  # here for the same reason as the range check above -- after the boot, every
  # task would fail at its first decision instead of one error costing nothing.
  if _AGENT_NAME.value == 'client_mobile_jev_v2' and not os.environ.get(
      'TYPESAFE_MM_ENDPOINT', ''
  ).strip():
    raise ValueError(
        'client_mobile_jev_v2 needs TYPESAFE_MM_ENDPOINT pointing at the'
        ' vision server (POST /v1/systemone).'
    )

  client = interface.AndroidEnvClient(base_url=_SERVER_URL.value)
  _wait_for_healthy_server(client, _HEALTH_TIMEOUT_SEC.value)

  client.reinitialize_suite(
      n_task_combinations=n_task_combinations,
      seed=_TASK_RANDOM_SEED.value,
      task_family=_SUITE_FAMILY.value,
      use_identical_params=use_identical_params,
  )
  task_types = _select_tasks(client, _TASKS.value, _TASK_INDEX_RANGE.value)

  agent = _get_agent(client, _SUITE_FAMILY.value)
  if _SUITE_FAMILY.value.startswith('miniwob'):
    # MiniWoB pages change quickly, don't need to wait for screen to stabilize.
    agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
  else:
    agent.transition_pause = None

  if _CHECKPOINT_DIR.value:
    checkpoint_dir = _CHECKPOINT_DIR.value
  else:
    checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)
  checkpointer = checkpointer_lib.IncrementalCheckpointer(checkpoint_dir)

  print(
      f'Starting eval with agent {_AGENT_NAME.value} and writing to'
      f' {checkpoint_dir}: {len(task_types)} tasks'
      + (
          f' from --task_index_range {_TASK_INDEX_RANGE.value}'
          if _TASK_INDEX_RANGE.value is not None
          else ''
      )
  )

  completed_tasks, failed_tasks = suite_utils._get_task_info(  # pylint: disable=protected-access
      checkpointer.load(fields=_METADATA_FIELDS)
  )
  episodes_metadata: list[dict[str, Any]] = []
  for task_type in task_types:
    msg = 'Running task: ' + task_type
    suite_utils._log_and_print(msg + '\n' + '=' * len(msg))  # pylint: disable=protected-access

    n_instances = client.get_suite_task_length(task_type)
    for i in range(n_instances):
      instance_name = (
          task_type + checkpointer_lib.INSTANCE_SEPARATOR + str(i)
      )
      # Resume semantics mirror suite_utils._run_task_suite: previously
      # completed instances are skipped, previously failed ones are retried.
      if instance_name in completed_tasks:
        episodes_metadata.extend(completed_tasks[instance_name])
      if instance_name in failed_tasks:
        episodes_metadata.extend(failed_tasks[instance_name])
      if (
          instance_name in completed_tasks
          and instance_name not in failed_tasks
      ):
        suite_utils._log_and_print(  # pylint: disable=protected-access
            'Skipping already processed task %s', instance_name
        )
        continue

      episode = _run_instance(
          client,
          agent,
          task_type,
          i,
          _instance_seed(
              _TASK_RANDOM_SEED.value, task_type, i, use_identical_params
          ),
          is_miniwob=_SUITE_FAMILY.value.startswith('miniwob'),
      )
      episode[constants.EpisodeConstants.AGENT_NAME] = agent.name
      episode[constants.EpisodeConstants.INSTANCE_ID] = i
      checkpointer.save_episodes([episode], instance_name)
      if _RAW_DUMPS.value:
        try:
          dump_dir = _dump_raw_episode(
              checkpoint_dir, instance_name, episode
          )
          print(f'Wrote raw dumps to {dump_dir}')
        except Exception as e:  # pylint: disable=broad-exception-caught
          # The .pkl.gz log is authoritative; a dump failure must not kill an
          # otherwise healthy evaluation run.
          print(f'Raw dump for {instance_name} failed (continuing): {e}')
      episodes_metadata.append({k: episode[k] for k in _METADATA_FIELDS})
      suite_utils.process_episodes(episodes_metadata, print_summary=True)
    print()

  print(
      f'Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value}'
      f' family. Wrote to {checkpoint_dir}.'
  )
  # Deliberately NOT calling client.close(): /close shuts down the server's
  # environment without stopping the emulator, leaving the container in a
  # "server alive, env dead" state that the watchdog does not recover (it
  # only restarts on QEMU death). Restart the container explicitly when
  # needed: ./server/restart_aw.sh 5000 (takes 5-10 minutes to boot).


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
