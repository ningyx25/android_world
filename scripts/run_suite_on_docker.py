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

"""Minimal client example for the Android environment server (Docker).

Prerequisites (from the repository root):

  # Build the image and start a container on host port 5000:
  ./scripts/run_container.sh 5000

  # ... wait for it to become healthy (5-10 minutes):
  curl http://localhost:5000/health

This script walks through the environment client's basic surface: health
check, reset, state, a single action, and one full task lifecycle. For
running a whole eval suite with checkpointing and result summaries, use
`run_on_docker.py` instead -- it is the Docker counterpart of `run.py`.

Note this script deliberately does NOT call client.close(): /close shuts down
the environment inside the container without stopping the emulator, leaving
the container unusable until it is restarted (./server/restart_aw.sh 5000,
which itself needs 5-10 minutes).
"""

import time

from android_world.env import interface
from android_world.env import json_action

BASE_URL = 'http://localhost:5000'
HEALTH_TIMEOUT_SEC = 900


def wait_for_healthy_server(client: interface.AndroidEnvClient) -> None:
  """Waits for the server's deep health probe to pass."""
  start = time.time()
  while not client.health():
    if time.time() - start > HEALTH_TIMEOUT_SEC:
      raise RuntimeError(
          f'Server at {client.base_url} did not become healthy within'
          f' {HEALTH_TIMEOUT_SEC}s. Check: docker logs aw_5000'
      )
    print('Environment is not healthy yet, waiting...')
    time.sleep(5)
  print('Environment server is healthy.')


def main() -> None:
  client = interface.AndroidEnvClient(base_url=BASE_URL)
  wait_for_healthy_server(client)

  res = client.reset(go_home=True)
  print(f'reset response: {res}')

  # get_state returns the server's own pixels + UI element list, so element
  # indices below are interpreted by /execute_action exactly as intended.
  state = client.get_state(wait_to_stabilize=True)
  print('Screen dimensions:', state.pixels.shape)
  print('UI elements:', len(state.ui_elements))

  res = client.execute_action(
      json_action.JSONAction(action_type='click', x=540, y=1200)
  )
  print(f'execute_action response: {res}')

  # Regenerate the suite with the same parameters run_on_docker.py uses.
  res = client.reinitialize_suite(
      n_task_combinations=1, seed=30, task_family='android_world'
  )
  print(f'reinitialize_suite response: {res}')

  task_list = client.get_suite_task_list(max_index=-1)
  print(f'{len(task_list)} tasks in the suite; first: {task_list[0]}')

  # A single task lifecycle. (run_on_docker.py does this for the whole
  # suite, with an agent choosing actions and checkpointing the results.)
  task_name = task_list[0]
  try:
    print(f'initialize_task: {client.initialize_task(task_name, 0)}')
    print(f'goal: {client.get_task_goal(task_name, 0)}')
    print(f'complexity: {client.get_task_complexity(task_name, 0)}')

    # Complete the task using your agent here, then score it. For example:
    #   agent = t3a.ClientT3A(client, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
    #   episode_runner.run_episode(goal=goal, agent=agent, ...)

    print(f'score: {client.get_task_score(task_name, 0)}')
    print(f'tear_down: {client.tear_down_task(task_name, 0)}')
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'Error running task {task_name}: {e}')


if __name__ == '__main__':
  main()
