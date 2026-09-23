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
import os
import time
import traceback
from typing import Any

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import constants
from android_world import episode_runner
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import t3a
from android_world.env import interface

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
  else:
    raise ValueError(
        f'Unknown agent for Docker mode: {_AGENT_NAME.value}. Client-backed'
        ' agents currently available: client_t3a. (The env-backed agents from'
        ' run.py -- human_agent, random_agent, m3a_*, t3a_*, seeact -- require'
        ' a local emulator and are not usable against the Docker server.)'
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


def _select_tasks(
    client: interface.AndroidEnvClient, tasks: list[str] | None
) -> list[str]:
  """Returns the suite's task keys, optionally filtered to `tasks`.

  Validation is done against the server's suite (not a local registry), since
  the server owns the task objects.

  Args:
    client: The environment client.
    tasks: Task keys to keep, or None to keep all.

  Returns:
    The task keys to run, in suite order.

  Raises:
    ValueError: If a requested task is not in the suite.
  """
  task_list = client.get_suite_task_list(max_index=-1)
  if tasks is None:
    return task_list
  for name in tasks:
    if name not in task_list:
      raise ValueError(
          f'Task {name} not found in the suite.'
          + suite_utils._suggest_keyword(name, task_list)  # pylint: disable=protected-access
      )
  return [name for name in task_list if name in set(tasks)]


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

  client = interface.AndroidEnvClient(base_url=_SERVER_URL.value)
  _wait_for_healthy_server(client, _HEALTH_TIMEOUT_SEC.value)

  client.reinitialize_suite(
      n_task_combinations=n_task_combinations,
      seed=_TASK_RANDOM_SEED.value,
      task_family=_SUITE_FAMILY.value,
      use_identical_params=use_identical_params,
  )
  task_types = _select_tasks(client, _TASKS.value)

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
      f' {checkpoint_dir}'
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
