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

"""Tests for the Docker suite runner (run_on_docker.py)."""

import contextlib
import json
import os
import shutil
import tempfile
from typing import Any
from unittest import mock

from absl import flags
from absl.testing import absltest
from android_world import checkpointer as checkpointer_lib
from android_world import constants
from android_world import episode_runner
from android_world.env import representation_utils
import numpy as np
from PIL import Image
import run_on_docker


class _FakeTaskSpec:
  """A task template's server-side metadata."""

  def __init__(
      self,
      task_type: str,
      n_instances: int,
      goal: str = 'do the thing',
      complexity: float = 1.0,
      start_on_home_screen: bool = True,
  ):
    self.task_type = task_type
    self.n_instances = n_instances
    self.goal = goal
    self.complexity = complexity
    self.start_on_home_screen = start_on_home_screen


class FakeAndroidEnvClient:
  """Stand-in for interface.AndroidEnvClient with the runner's surface."""

  def __init__(
      self,
      task_specs: list[_FakeTaskSpec] | None = None,
      healthy: bool = True,
      fail_initialize_for: set[str] | None = None,
      fail_score_for: set[str] | None = None,
  ):
    self.base_url = 'http://fake-server:5000'
    self._specs = {
        spec.task_type: spec
        for spec in (
            task_specs
            if task_specs is not None
            else [_FakeTaskSpec('FakeTaskA', 2), _FakeTaskSpec('FakeTaskB', 1)]
        )
    }
    self._healthy = healthy
    self._fail_initialize_for = fail_initialize_for or set()
    self._fail_score_for = fail_score_for or set()

    self.calls: list[tuple[Any, ...]] = []
    self.reinitialize_kwargs: dict[str, Any] | None = None
    self.saved_episodes: dict[str, list[dict[str, Any]]] = {}
    self.closed = False

  # --- lifecycle -----------------------------------------------------------
  def health(self) -> bool:
    self.calls.append(('health',))
    return self._healthy

  def reinitialize_suite(
      self,
      n_task_combinations: int = 2,
      seed: int = 42,
      task_family: str = 'android_world',
      use_identical_params: bool = False,
  ):
    self.calls.append(('reinitialize_suite',))
    self.reinitialize_kwargs = {
        'n_task_combinations': n_task_combinations,
        'seed': seed,
        'task_family': task_family,
        'use_identical_params': use_identical_params,
    }
    return mock.MagicMock()

  def close(self) -> None:
    self.closed = True

  # --- suite ---------------------------------------------------------------
  def get_suite_task_list(self, max_index: int) -> list[str]:
    self.calls.append(('get_suite_task_list', max_index))
    return list(self._specs.keys())

  def get_suite_task_length(self, task_type: str) -> int:
    self.calls.append(('get_suite_task_length', task_type))
    return self._specs[task_type].n_instances

  # --- task ----------------------------------------------------------------
  def initialize_task(self, task_type: str, task_idx: int):
    self.calls.append(('initialize_task', task_type, task_idx))
    if task_type in self._fail_initialize_for:
      raise RuntimeError(f'initialize failed for {task_type}')
    return mock.MagicMock()

  def get_task_goal(self, task_type: str, task_idx: int) -> str:
    self.calls.append(('get_task_goal', task_type, task_idx))
    return self._specs[task_type].goal

  def get_task_complexity(self, task_type: str, task_idx: int) -> float:
    self.calls.append(('get_task_complexity', task_type, task_idx))
    return self._specs[task_type].complexity

  def start_on_home_screen(self, task_type: str, task_idx: int) -> bool:
    self.calls.append(('start_on_home_screen', task_type, task_idx))
    return self._specs[task_type].start_on_home_screen

  def get_task_score(self, task_type: str, task_idx: int) -> float:
    self.calls.append(('get_task_score', task_type, task_idx))
    if task_type in self._fail_score_for:
      raise RuntimeError(f'score failed for {task_type}')
    return 1.0

  def tear_down_task(self, task_type: str, task_idx: int):
    self.calls.append(('tear_down_task', task_type, task_idx))
    return mock.MagicMock()

  def is_miniwob_episode_terminated(self) -> bool:
    self.calls.append(('is_miniwob_episode_terminated',))
    return False


class _RecordingCheckpointer(checkpointer_lib.Checkpointer):
  """In-memory checkpointer that records saves and serves resumable data."""

  def __init__(self, existing: list[dict[str, Any]] | None = None):
    self.saved: dict[str, list[dict[str, Any]]] = {}
    self._existing = existing or []

  def save_episodes(self, task_episodes, task_name):
    self.saved[task_name] = task_episodes

  def load(self, fields=None):
    if fields is None:
      return list(self._existing)
    return [{k: episode[k] for k in fields} for episode in self._existing]


def _make_agent(done: bool = True, fail_at_step: int | None = None):
  """Builds a minimal client-shaped agent.

  Args:
    done: Whether step() reports the task as done.
    fail_at_step: If set, the agent raises on that step index (0-based).

  Returns:
    A mock agent whose step() records calls.
  """
  agent = mock.MagicMock()
  agent.name = 'fake_agent'
  agent.transition_pause = 1.0
  steps: list[int] = []

  def _step(goal: str) -> Any:
    index = len(steps)
    steps.append(index)
    if fail_at_step is not None and index == fail_at_step:
      raise RuntimeError('agent exploded')
    result = mock.MagicMock()
    result.done = done
    result.data = {'action': f'action_{index}'}
    return result

  agent.step.side_effect = _step
  # ClientInteractingAgent has no `.env`; ensure the runner never needs one.
  del agent.env
  return agent


@contextlib.contextmanager
def _patched_runner(
    client: FakeAndroidEnvClient,
    agent: Any,
    checkpointer: checkpointer_lib.Checkpointer,
):
  """Runs run_on_docker._main against the given client/agent/checkpointer."""
  with mock.patch.object(
      run_on_docker.interface, 'AndroidEnvClient', return_value=client
  ), mock.patch.object(
      run_on_docker, '_get_agent', return_value=agent
  ), mock.patch.object(
      run_on_docker,
      '_wait_for_healthy_server',
      side_effect=lambda c, t: None,
  ), mock.patch.object(
      run_on_docker.checkpointer_lib,
      'IncrementalCheckpointer',
      return_value=checkpointer,
  ):
    yield


@contextlib.contextmanager
def _reset_flags(**overrides: Any):
  """Pins the runner's flags for a test and restores them afterwards."""
  flags.FLAGS.mark_as_parsed()  # Flags are normally parsed by app.run().
  values = {
      'server_url': 'http://fake-server:5000',
      'suite_family': 'android_world',
      'task_random_seed': 30,
      'tasks': None,
      'task_index_range': None,
      'n_task_combinations': 1,
      'checkpoint_dir': '',
      'output_path': '',
      'agent_name': 'client_t3a',
      'fixed_task_seed': False,
      'health_timeout_sec': 900,
      # Off by default so tests never write dumps into the working directory;
      # RawDumpTest turns it on with a temporary checkpoint dir.
      'raw_dumps': False,
  }
  values.update(overrides)
  saved = {name: flags.FLAGS[name].value for name in values}
  try:
    for name, value in values.items():
      flags.FLAGS[name].value = value
    yield
  finally:
    for name, value in saved.items():
      flags.FLAGS[name].value = value


class SelectTasksTest(absltest.TestCase):

  def test_returns_all_tasks_when_none_requested(self):
    client = FakeAndroidEnvClient()
    self.assertEqual(
        run_on_docker._select_tasks(client, None),
        ['FakeTaskA', 'FakeTaskB'],
    )

  def test_filters_to_requested_subset_preserving_suite_order(self):
    client = FakeAndroidEnvClient()
    self.assertEqual(
        run_on_docker._select_tasks(client, ['FakeTaskB']), ['FakeTaskB']
    )
    self.assertEqual(
        run_on_docker._select_tasks(client, ['FakeTaskB', 'FakeTaskA']),
        ['FakeTaskA', 'FakeTaskB'],
    )

  def test_unknown_task_raises_with_suggestion(self):
    client = FakeAndroidEnvClient()
    with self.assertRaisesRegex(
        ValueError, 'not found in the suite.*FakeTaskA'
    ):
      run_on_docker._select_tasks(client, ['FakeTasA'])


class TaskIndexRangeTest(absltest.TestCase):

  def test_parses_bounds(self):
    self.assertEqual(run_on_docker._parse_task_index_range('1-58'), (1, 58))
    self.assertEqual(run_on_docker._parse_task_index_range('2-2'), (2, 2))
    self.assertEqual(run_on_docker._parse_task_index_range(' 3 - 4 '), (3, 4))

  def test_malformed_range_raises(self):
    for spec in ('', 'abc', '1', '-5', '1-', '1-2-3'):
      with self.assertRaisesRegex(ValueError, 'expected "START-END"'):
        run_on_docker._parse_task_index_range(spec)

  def test_zero_start_raises(self):
    with self.assertRaisesRegex(ValueError, '1-based'):
      run_on_docker._parse_task_index_range('0-3')

  def test_reversed_range_raises(self):
    with self.assertRaisesRegex(ValueError, 'must not exceed END'):
      run_on_docker._parse_task_index_range('58-1')

  def test_slices_suite_positions(self):
    suite = ['a', 'b', 'c', 'd']
    self.assertEqual(
        run_on_docker._slice_task_index_range(suite, '2-3'), ['b', 'c']
    )
    self.assertEqual(
        run_on_docker._slice_task_index_range(suite, '4-4'), ['d']
    )
    self.assertEqual(run_on_docker._slice_task_index_range(suite, None), suite)

  def test_end_past_suite_raises(self):
    with self.assertRaisesRegex(ValueError, 'out of bounds.*4 tasks'):
      run_on_docker._slice_task_index_range(['a', 'b', 'c', 'd'], '1-5')


class InstanceSeedTest(absltest.TestCase):

  def test_matches_suite_utils_seeding(self):
    """The recorded seed must match what create_suite would have used."""
    import hashlib  # pylint: disable=g-import-not-at-top

    for seed, name, index in [(30, 'FakeTaskA', 0), (30, 'FakeTaskA', 3)]:
      expected = int(
          hashlib.sha256(f'{seed}_{name}_{index}'.encode()).hexdigest(), 16
      ) % (2**32)
      self.assertEqual(
          run_on_docker._instance_seed(seed, name, index, False), expected
      )

  def test_identical_params_uses_instance_zero_seed(self):
    self.assertEqual(
        run_on_docker._instance_seed(30, 'FakeTaskA', 2, True),
        run_on_docker._instance_seed(30, 'FakeTaskA', 0, True),
    )

  def test_no_seed_returns_none(self):
    self.assertIsNone(run_on_docker._instance_seed(None, 'FakeTaskA', 0, False))


class BuildEpisodeTest(absltest.TestCase):

  def _episode(self, done: bool) -> episode_runner.EpisodeResult:
    return episode_runner.EpisodeResult(
        done=done,
        step_data={constants.STEP_NUMBER: [0, 1]},
        aux_data={'extra': 1},
    )

  def test_success_record_has_all_expected_fields(self):
    record = run_on_docker._build_episode(
        'FakeTaskA',
        'goal text',
        self._episode(done=True),
        task_score=1.0,
        run_time=1.5,
        seed=123,
    )
    for field in (
        constants.EpisodeConstants.GOAL,
        constants.EpisodeConstants.TASK_TEMPLATE,
        constants.EpisodeConstants.EPISODE_DATA,
        constants.EpisodeConstants.IS_SUCCESSFUL,
        constants.EpisodeConstants.RUN_TIME,
        constants.EpisodeConstants.FINISH_DTIME,
        constants.EpisodeConstants.EPISODE_LENGTH,
        constants.EpisodeConstants.AUX_DATA,
        constants.EpisodeConstants.SCREEN_CONFIG,
        constants.EpisodeConstants.EXCEPTION_INFO,
        constants.EpisodeConstants.SEED,
    ):
      self.assertIn(field, record)
    self.assertEqual(record[constants.EpisodeConstants.IS_SUCCESSFUL], 1.0)
    self.assertEqual(record[constants.EpisodeConstants.EPISODE_LENGTH], 2)
    self.assertIsNone(record[constants.EpisodeConstants.EXCEPTION_INFO])
    self.assertEqual(record[constants.EpisodeConstants.SEED], 123)

  def test_not_done_scores_zero_even_when_server_says_successful(self):
    """Mirrors suite_utils._run_task: no completion means no credit."""
    record = run_on_docker._build_episode(
        'FakeTaskA',
        'goal text',
        self._episode(done=False),
        task_score=1.0,
        run_time=1.5,
        seed=1,
    )
    self.assertEqual(record[constants.EpisodeConstants.IS_SUCCESSFUL], 0.0)


class RunInstanceTest(absltest.TestCase):

  def test_initialize_precedes_goal_fetch(self):
    """MiniWoB goals are invalid before initialization; order matters."""
    client = FakeAndroidEnvClient()
    agent = _make_agent(done=True)
    run_on_docker._run_instance(client, agent, 'FakeTaskA', 0, seed=1, is_miniwob=False)

    names = [call[0] for call in client.calls]
    self.assertLess(
        names.index('initialize_task'), names.index('get_task_goal')
    )

  def test_successful_instance_tears_down_and_returns_record(self):
    client = FakeAndroidEnvClient()
    agent = _make_agent(done=True)
    record = run_on_docker._run_instance(
        client, agent, 'FakeTaskA', 0, seed=7, is_miniwob=False
    )

    self.assertEqual(record[constants.EpisodeConstants.IS_SUCCESSFUL], 1.0)
    self.assertIsNone(record[constants.EpisodeConstants.EXCEPTION_INFO])
    self.assertIn(('tear_down_task', 'FakeTaskA', 0), client.calls)

  def test_initialize_failure_returns_failed_result_without_teardown(self):
    client = FakeAndroidEnvClient(fail_initialize_for={'FakeTaskA'})
    agent = _make_agent(done=True)
    record = run_on_docker._run_instance(
        client, agent, 'FakeTaskA', 0, seed=1, is_miniwob=False
    )

    self.assertIsNotNone(record[constants.EpisodeConstants.EXCEPTION_INFO])
    self.assertTrue(np.isnan(record[constants.EpisodeConstants.IS_SUCCESSFUL]))
    self.assertNotIn(('tear_down_task', 'FakeTaskA', 0), client.calls)

  def test_score_failure_is_an_episode_failure_not_a_zero(self):
    """A scoring error must not be silently recorded as a completed task."""
    client = FakeAndroidEnvClient(fail_score_for={'FakeTaskA'})
    agent = _make_agent(done=True)
    record = run_on_docker._run_instance(
        client, agent, 'FakeTaskA', 0, seed=1, is_miniwob=False
    )

    self.assertIsNotNone(record[constants.EpisodeConstants.EXCEPTION_INFO])

  def test_agent_crash_returns_failed_result(self):
    client = FakeAndroidEnvClient()
    agent = _make_agent(done=False, fail_at_step=0)
    record = run_on_docker._run_instance(
        client, agent, 'FakeTaskA', 0, seed=1, is_miniwob=False
    )

    self.assertIsNotNone(record[constants.EpisodeConstants.EXCEPTION_INFO])

  def test_miniwob_installs_termination_check(self):
    client = FakeAndroidEnvClient()
    agent = _make_agent(done=False, fail_at_step=1)
    run_on_docker._run_instance(
        client, agent, 'FakeTaskA', 0, seed=1, is_miniwob=True
    )

    self.assertIn(('is_miniwob_episode_terminated',), client.calls)

  def test_non_miniwob_does_not_poll_termination(self):
    client = FakeAndroidEnvClient()
    agent = _make_agent(done=False, fail_at_step=1)
    run_on_docker._run_instance(
        client, agent, 'FakeTaskA', 0, seed=1, is_miniwob=False
    )

    self.assertNotIn(('is_miniwob_episode_terminated',), client.calls)

  def test_max_steps_scales_with_complexity(self):
    client = FakeAndroidEnvClient(
        task_specs=[_FakeTaskSpec('FakeTaskA', 1, complexity=1.5)]
    )
    # int(10 * 1.5) = 15 steps allowed; step index 15 must never be reached.
    agent = _make_agent(done=False, fail_at_step=15)
    run_on_docker._run_instance(
        client, agent, 'FakeTaskA', 0, seed=1, is_miniwob=False
    )
    self.assertLen(agent.step.call_args_list, 15)


class WaitForHealthyServerTest(absltest.TestCase):

  def test_returns_immediately_when_healthy(self):
    client = FakeAndroidEnvClient(healthy=True)
    run_on_docker._wait_for_healthy_server(client, timeout_sec=10)
    self.assertEqual(client.calls, [('health',)])

  def test_timeout_raises(self):
    client = FakeAndroidEnvClient(healthy=False)
    with mock.patch.object(run_on_docker.time, 'sleep', return_value=None):
      with self.assertRaisesRegex(RuntimeError, 'not healthy'):
        run_on_docker._wait_for_healthy_server(client, timeout_sec=0)


class MainFlowTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

  def test_runs_each_instance_builds_metadata_and_saves(self):
    client = FakeAndroidEnvClient(
        task_specs=[_FakeTaskSpec('FakeTaskA', 2), _FakeTaskSpec('FakeTaskB', 1)]
    )
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()

    with _reset_flags(), _patched_runner(client, agent, checkpointer):
      run_on_docker._main()

    self.assertEqual(
        set(checkpointer.saved.keys()), {'FakeTaskA_0', 'FakeTaskA_1', 'FakeTaskB_0'}
    )
    for episodes in checkpointer.saved.values():
      episode = episodes[0]
      self.assertEqual(
          episode[constants.EpisodeConstants.AGENT_NAME], 'fake_agent'
      )
      self.assertIn(constants.EpisodeConstants.INSTANCE_ID, episode)
      self.assertEqual(episode[constants.EpisodeConstants.IS_SUCCESSFUL], 1.0)

    # Suite was re-created with the flags' parameters.
    self.assertEqual(
        client.reinitialize_kwargs,
        {
            'n_task_combinations': 1,
            'seed': 30,
            'task_family': 'android_world',
            'use_identical_params': False,
        },
    )

  def test_does_not_close_the_client(self):
    """Closing the server's env bricks the container without a watchdog kick."""
    client = FakeAndroidEnvClient()
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()

    with _reset_flags(), _patched_runner(client, agent, checkpointer):
      run_on_docker._main()

    self.assertFalse(client.closed)

  def test_tasks_flag_limits_run(self):
    client = FakeAndroidEnvClient(
        task_specs=[_FakeTaskSpec('FakeTaskA', 2), _FakeTaskSpec('FakeTaskB', 1)]
    )
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()

    with _reset_flags(tasks=['FakeTaskB']), _patched_runner(
        client, agent, checkpointer
    ):
      run_on_docker._main()

    self.assertEqual(set(checkpointer.saved.keys()), {'FakeTaskB_0'})

  def test_task_index_range_limits_run(self):
    client = FakeAndroidEnvClient(
        task_specs=[
            _FakeTaskSpec('FakeTaskA', 1),
            _FakeTaskSpec('FakeTaskB', 1),
            _FakeTaskSpec('FakeTaskC', 1),
        ]
    )
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()

    with _reset_flags(task_index_range='2-3'), _patched_runner(
        client, agent, checkpointer
    ):
      run_on_docker._main()

    self.assertEqual(
        set(checkpointer.saved.keys()), {'FakeTaskB_0', 'FakeTaskC_0'}
    )

  def test_task_index_range_counts_over_the_suite_not_the_tasks_filter(self):
    client = FakeAndroidEnvClient(
        task_specs=[
            _FakeTaskSpec('FakeTaskA', 1),
            _FakeTaskSpec('FakeTaskB', 1),
            _FakeTaskSpec('FakeTaskC', 1),
        ]
    )
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()

    # Suite positions 1-2 are FakeTaskA and FakeTaskB, and only FakeTaskA is
    # also in --tasks -- so it runs even though it is not first there.
    with _reset_flags(
        tasks=['FakeTaskC', 'FakeTaskA'], task_index_range='1-2'
    ), _patched_runner(client, agent, checkpointer):
      run_on_docker._main()

    self.assertEqual(set(checkpointer.saved.keys()), {'FakeTaskA_0'})

  def test_tasks_and_range_without_overlap_raises(self):
    client = FakeAndroidEnvClient(
        task_specs=[_FakeTaskSpec('FakeTaskA', 1), _FakeTaskSpec('FakeTaskB', 1)]
    )
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()

    with _reset_flags(
        tasks=['FakeTaskB'], task_index_range='1-1'
    ), _patched_runner(client, agent, checkpointer):
      with self.assertRaisesRegex(ValueError, 'select no tasks in common'):
        run_on_docker._main()

  def test_out_of_bounds_range_raises_before_running_anything(self):
    client = FakeAndroidEnvClient(task_specs=[_FakeTaskSpec('FakeTaskA', 1)])
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()

    with _reset_flags(task_index_range='1-58'), _patched_runner(
        client, agent, checkpointer
    ):
      with self.assertRaisesRegex(ValueError, 'out of bounds'):
        run_on_docker._main()

    self.assertEmpty(checkpointer.saved)

  def test_malformed_range_fails_before_the_health_wait(self):
    client = FakeAndroidEnvClient(task_specs=[_FakeTaskSpec('FakeTaskA', 1)])
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()

    with _reset_flags(task_index_range='1..58'), _patched_runner(
        client, agent, checkpointer
    ):
      with self.assertRaisesRegex(ValueError, 'expected "START-END"'):
        run_on_docker._main()

    # The typo was caught before the client was even polled for health.
    self.assertEmpty([call for call in client.calls if call[0] == 'health'])
    self.assertEmpty(checkpointer.saved)

  def test_fixed_task_seed_forwarded_to_reinitialize(self):
    client = FakeAndroidEnvClient(task_specs=[_FakeTaskSpec('FakeTaskA', 2)])
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()

    with _reset_flags(fixed_task_seed=True), _patched_runner(
        client, agent, checkpointer
    ):
      run_on_docker._main()

    self.assertTrue(client.reinitialize_kwargs['use_identical_params'])

  def test_completed_instances_are_skipped_on_resume(self):
    client = FakeAndroidEnvClient(task_specs=[_FakeTaskSpec('FakeTaskA', 2)])
    agent = _make_agent(done=True)
    completed_episode = {
        constants.EpisodeConstants.GOAL: 'old goal',
        constants.EpisodeConstants.TASK_TEMPLATE: 'FakeTaskA',
        constants.EpisodeConstants.INSTANCE_ID: 0,
        constants.EpisodeConstants.IS_SUCCESSFUL: 1.0,
        constants.EpisodeConstants.EPISODE_LENGTH: 3,
        constants.EpisodeConstants.RUN_TIME: 2.0,
        constants.EpisodeConstants.EXCEPTION_INFO: None,
        constants.EpisodeConstants.AUX_DATA: None,
    }
    checkpointer = _RecordingCheckpointer(existing=[completed_episode])

    with _reset_flags(), _patched_runner(client, agent, checkpointer):
      run_on_docker._main()

    # Instance 0 was already complete, so only instance 1 ran.
    self.assertEqual(set(checkpointer.saved.keys()), {'FakeTaskA_1'})
    ran = [call[2] for call in client.calls if call[0] == 'initialize_task']
    self.assertEqual(ran, [1])

  def test_init_failure_does_not_abort_the_suite(self):
    client = FakeAndroidEnvClient(
        task_specs=[_FakeTaskSpec('FakeTaskA', 1), _FakeTaskSpec('FakeTaskB', 1)],
        fail_initialize_for={'FakeTaskA'},
    )
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()

    with _reset_flags(), _patched_runner(client, agent, checkpointer):
      run_on_docker._main()

    self.assertEqual(set(checkpointer.saved.keys()), {'FakeTaskA_0', 'FakeTaskB_0'})
    failed = checkpointer.saved['FakeTaskA_0'][0]
    self.assertIsNotNone(failed[constants.EpisodeConstants.EXCEPTION_INFO])
    succeeded = checkpointer.saved['FakeTaskB_0'][0]
    self.assertIsNone(succeeded[constants.EpisodeConstants.EXCEPTION_INFO])

  def test_episode_length_and_seed_recorded(self):
    client = FakeAndroidEnvClient(task_specs=[_FakeTaskSpec('FakeTaskA', 1)])
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()

    with _reset_flags(task_random_seed=30), _patched_runner(
        client, agent, checkpointer
    ):
      run_on_docker._main()

    episode = checkpointer.saved['FakeTaskA_0'][0]
    self.assertEqual(episode[constants.EpisodeConstants.EPISODE_LENGTH], 1)
    self.assertEqual(
        episode[constants.EpisodeConstants.SEED],
        run_on_docker._instance_seed(30, 'FakeTaskA', 0, False),
    )


class CheckpointDirTest(absltest.TestCase):

  def test_creates_run_directory_under_output_path(self):
    with tempfile.TemporaryDirectory() as tmp:
      output_path = os.path.join(tmp, 'runs')
      client = FakeAndroidEnvClient(task_specs=[_FakeTaskSpec('FakeTaskA', 1)])
      agent = _make_agent(done=True)

      with _reset_flags(checkpoint_dir='', output_path=output_path):
        with mock.patch.object(
            run_on_docker.interface, 'AndroidEnvClient', return_value=client
        ), mock.patch.object(
            run_on_docker, '_get_agent', return_value=agent
        ), mock.patch.object(
            run_on_docker, '_wait_for_healthy_server', return_value=None
        ):
          run_on_docker._main()

      run_dirs = [
          name for name in os.listdir(output_path) if name.startswith('run_')
      ]
      self.assertLen(run_dirs, 1)
      saved = os.listdir(os.path.join(output_path, run_dirs[0]))
      self.assertEqual(saved, ['FakeTaskA_0.pkl.gz'])

  def test_explicit_checkpoint_dir_is_used(self):
    with tempfile.TemporaryDirectory() as tmp:
      client = FakeAndroidEnvClient(task_specs=[_FakeTaskSpec('FakeTaskA', 1)])
      agent = _make_agent(done=True)

      with _reset_flags(checkpoint_dir=tmp):
        with mock.patch.object(
            run_on_docker.interface, 'AndroidEnvClient', return_value=client
        ), mock.patch.object(
            run_on_docker, '_get_agent', return_value=agent
        ), mock.patch.object(
            run_on_docker, '_wait_for_healthy_server', return_value=None
        ):
          run_on_docker._main()

      self.assertEqual(os.listdir(tmp), ['FakeTaskA_0.pkl.gz'])


class RawDumpTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

  def _episode(self) -> dict[str, Any]:
    ui_element = representation_utils.UIElement(
        text='Wi-Fi',
        class_name='android.widget.TextView',
        bbox_pixels=representation_utils.BoundingBox(0, 10, 0, 5),
        is_visible=True,
    )
    step_data = {
        constants.STEP_NUMBER: [0, 1],
        'before_screenshot': [np.zeros((4, 4, 3), dtype=np.uint8), None],
        'after_screenshot': [np.full((4, 4, 3), 255, dtype=np.uint8), None],
        'before_element_list': [[ui_element], [ui_element]],
        'action_prompt': [{'state': {'goal': 'g'}}, None],
        'action_output': ['{"answers": {}}', None],
        'mobile_jev': [{'status': 'action'}, {'status': 'done'}],
        'summary': ['selected tap', 'done'],
    }
    episode_constants = constants.EpisodeConstants
    return {
        episode_constants.EPISODE_DATA: step_data,
        episode_constants.GOAL: 'do the thing',
        episode_constants.TASK_TEMPLATE: 'FakeTaskA',
        episode_constants.AGENT_NAME: 'client_mobile_jev',
        episode_constants.IS_SUCCESSFUL: 1.0,
        episode_constants.RUN_TIME: 1.5,
        episode_constants.EPISODE_LENGTH: 2,
        episode_constants.SEED: 42,
        episode_constants.EXCEPTION_INFO: None,
    }

  def test_step_frame_prefers_before_then_after(self):
    before = np.zeros((4, 4, 3), dtype=np.uint8)
    after = np.full((4, 4, 3), 255, dtype=np.uint8)
    step_data = {
        'before_screenshot': [before],
        'after_screenshot': [after],
    }

    self.assertIs(run_on_docker._step_frame(step_data, 0), before)
    step_data['before_screenshot'] = [None]
    self.assertIs(run_on_docker._step_frame(step_data, 0), after)
    step_data['after_screenshot'] = [None]
    self.assertIsNone(run_on_docker._step_frame(step_data, 0))

  def test_writes_one_image_and_json_per_step(self):
    dump_dir = run_on_docker._dump_raw_episode(
        self._tmp, 'FakeTaskA_0', self._episode()
    )

    self.assertEqual(dump_dir, os.path.join(self._tmp, 'raw_dumps',
                                            'FakeTaskA_0'))
    self.assertEqual(
        sorted(os.listdir(dump_dir)),
        ['episode.json', 'step_001.json', 'step_001.png', 'step_002.json',
         'step_002.png'],
    )
    with open(os.path.join(dump_dir, 'step_001.json')) as f:
      payload = json.load(f)
    self.assertEqual(payload['step'], 1)
    self.assertEqual(payload['goal'], 'do the thing')
    self.assertEqual(payload['screenshot'], 'step_001.png')
    self.assertNotIn('screenshot_from_step', payload)
    self.assertNotIn('screenshots', payload)
    self.assertEqual(payload['action_output'], '{"answers": {}}')
    self.assertEqual(payload['mobile_jev']['status'], 'action')
    # Element lists are stored as plain dicts, not UIElement dataclasses.
    self.assertEqual(payload['before_element_list'][0]['text'], 'Wi-Fi')
    self.assertNotIsInstance(
        payload['before_element_list'][0], representation_utils.UIElement
    )
    # The before frame wins over the after frame: step 1's before frame is
    # black, its after frame white.
    with Image.open(os.path.join(dump_dir, 'step_001.png')) as image:
      self.assertEqual(image.size, (4, 4))
      self.assertEqual(image.convert('RGB').getpixel((0, 0)), (0, 0, 0))
    # The second step captured no frame, so it reuses the first step's, which
    # is still the current screen -- its JSON says where the frame came from.
    with open(os.path.join(dump_dir, 'step_002.json')) as f:
      second = json.load(f)
    self.assertEqual(second['screenshot'], 'step_002.png')
    self.assertEqual(second['screenshot_from_step'], 1)
    with Image.open(os.path.join(dump_dir, 'step_002.png')) as image:
      self.assertEqual(image.convert('RGB').getpixel((0, 0)), (0, 0, 0))
    with open(os.path.join(dump_dir, 'episode.json')) as f:
      summary = json.load(f)
    self.assertEqual(summary['instance'], 'FakeTaskA_0')
    self.assertEqual(summary['agent'], 'client_mobile_jev')
    self.assertEqual(summary['is_successful'], 1.0)
    self.assertEqual(summary['episode_length'], 2)

  def test_failed_episode_without_step_data_still_writes_summary(self):
    episode = {
        constants.EpisodeConstants.GOAL: 'do the thing',
        constants.EpisodeConstants.EXCEPTION_INFO: 'boom',
    }

    dump_dir = run_on_docker._dump_raw_episode(
        self._tmp, 'FakeTaskA_0', episode
    )

    self.assertEqual(os.listdir(dump_dir), ['episode.json'])

  def test_main_writes_dumps_next_to_the_pickle_log(self):
    client = FakeAndroidEnvClient(task_specs=[_FakeTaskSpec('FakeTaskA', 1)])
    agent = _make_agent(done=True)
    # The real checkpointer writes the .pkl.gz so both siblings are asserted.
    checkpointer = checkpointer_lib.IncrementalCheckpointer(self._tmp)

    with _reset_flags(
        checkpoint_dir=self._tmp, raw_dumps=True
    ), _patched_runner(client, agent, checkpointer):
      run_on_docker._main()

    self.assertEqual(
        sorted(os.listdir(self._tmp)),
        ['FakeTaskA_0.pkl.gz', 'raw_dumps'],
    )
    step_path = os.path.join(
        self._tmp, 'raw_dumps', 'FakeTaskA_0', 'step_001.json'
    )
    with open(step_path) as f:
      payload = json.load(f)
    self.assertEqual(payload['action'], 'action_0')
    self.assertTrue(
        os.path.exists(os.path.join(self._tmp, 'raw_dumps', 'FakeTaskA_0',
                                    'episode.json'))
    )

  def test_dump_failure_does_not_abort_the_run(self):
    client = FakeAndroidEnvClient(task_specs=[_FakeTaskSpec('FakeTaskA', 1)])
    agent = _make_agent(done=True)
    checkpointer = _RecordingCheckpointer()
    dump_failure = mock.patch.object(
        run_on_docker,
        '_dump_raw_episode',
        side_effect=RuntimeError('disk full'),
    )

    with _reset_flags(
        checkpoint_dir=self._tmp, raw_dumps=True
    ), _patched_runner(client, agent, checkpointer), dump_failure:
      run_on_docker._main()

    self.assertEqual(set(checkpointer.saved.keys()), {'FakeTaskA_0'})


if __name__ == '__main__':
  absltest.main()
