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

"""Tests for readiness gating and /health (all external calls mocked)."""

import asyncio
import json
import sys
import types
from unittest import mock

from absl.testing import absltest

# suite_utils imports `thefuzz`, which is declared in requirements.txt but not
# in pyproject.toml (the .venv is built from the latter). Stub it so importing
# android_server does not need an extra install in unit tests.
try:
  import thefuzz  # pylint: disable=unused-import  # noqa: F401
except ImportError:
  _thefuzz = types.ModuleType('thefuzz')
  _thefuzz_process = types.ModuleType('thefuzz.process')
  _thefuzz_process.extractOne = lambda *args, **kwargs: (None, 0)
  _thefuzz.process = _thefuzz_process
  sys.modules.setdefault('thefuzz', _thefuzz)
  sys.modules.setdefault('thefuzz.process', _thefuzz_process)

import fastapi  # pylint: disable=g-import-not-at-top

from server import android_server  # pylint: disable=g-import-not-at-top


def _fake_request(**overrides):
  values = dict(
      ready=False,
      init_error=None,
      app_android_env=None,
      suite=None,
      task_registry=None,
      adb_path='/opt/android/platform-tools/adb',
      init_task=None,
  )
  values.update(overrides)
  request = mock.Mock()
  request.app.state = types.SimpleNamespace(**values)
  return request


def _healthy_probe():
  return {
      'device': 'emulator-5554', 'qemu': True, 'adb_state': 'device',
      'boot_completed': True, 'healthy': True, 'reason': 'ok',
  }


class HealthTest(absltest.TestCase):

  def test_health_reports_initializing_before_ready(self):
    response = asyncio.run(android_server.health(_fake_request()))
    self.assertEqual(response.status_code, 503)
    body = json.loads(response.body)
    self.assertEqual(body['status'], 'error')
    self.assertEqual(body['reason'], 'initializing')
    self.assertEqual(response.headers['retry-after'], '30')

  def test_health_qemu_missing_does_not_start_recovery_thread(self):
    # /health delegates the decision to the recovery gate; a missing QEMU must
    # never start the adb-only recovery thread (the watchdog owns that path).
    probe = dict(_healthy_probe(), healthy=False, reason='qemu_missing')
    with mock.patch.object(
        android_server.health_probe, 'probe_device', return_value=probe
    ), mock.patch.object(
        android_server, '_unhealthy_since', None
    ), mock.patch.object(
        android_server, 'threading'
    ) as threading_mock:
      response = asyncio.run(
          android_server.health(_fake_request(ready=True))
      )
    self.assertEqual(response.status_code, 503)
    self.assertEqual(json.loads(response.body)['reason'], 'qemu_missing')
    threading_mock.Thread.assert_not_called()

  def test_health_adb_offline_schedules_soft_recovery(self):
    probe = dict(_healthy_probe(), healthy=False, reason='adb_offline')
    with mock.patch.object(
        android_server.health_probe, 'probe_device', return_value=probe
    ), mock.patch.object(
        android_server, '_request_soft_recovery'
    ) as recovery_mock:
      response = asyncio.run(
          android_server.health(_fake_request(ready=True))
      )
    self.assertEqual(response.status_code, 503)
    self.assertEqual(response.headers['retry-after'], '30')
    recovery_mock.assert_called_once_with('adb_offline')

  def test_health_returns_success_when_ready_and_healthy(self):
    with mock.patch.object(
        android_server.health_probe, 'probe_device',
        return_value=_healthy_probe(),
    ):
      response = asyncio.run(
          android_server.health(_fake_request(ready=True))
      )
    self.assertEqual(response, {'status': 'success'})


class DependencyGuardTest(absltest.TestCase):

  def test_env_dependency_raises_503_while_initializing(self):
    with self.assertRaises(fastapi.HTTPException) as ctx:
      android_server.get_app_android_env(_fake_request())
    self.assertEqual(ctx.exception.status_code, 503)

  def test_suite_dependency_raises_503_while_initializing(self):
    with self.assertRaises(fastapi.HTTPException) as ctx:
      android_server.get_app_suite(_fake_request())
    self.assertEqual(ctx.exception.status_code, 503)

  def test_dependencies_return_values_once_ready(self):
    env, suite = object(), object()
    self.assertIs(
        android_server.get_app_android_env(
            _fake_request(app_android_env=env)
        ),
        env,
    )
    self.assertIs(
        android_server.get_app_suite(_fake_request(suite=suite)), suite
    )


class InitializeAsyncTest(absltest.TestCase):

  def _state(self):
    return types.SimpleNamespace(
        adb_path='/adb', ready=False, init_error=None,
        app_android_env=None, suite=None, task_registry=None, init_task=None,
    )

  def test_publishes_env_suite_and_registry_atomically(self):
    app = mock.Mock()
    app.state = self._state()
    env = object()
    registry = mock.Mock()
    registry.get_registry.return_value = {'task': []}
    suite = object()

    def immediate(func, *args, **kwargs):
      return func(*args, **kwargs)

    with mock.patch.object(
        android_server.asyncio, 'to_thread',
        new=mock.AsyncMock(side_effect=immediate),
    ), mock.patch.object(
        android_server.env_launcher, 'load_and_setup_env',
        return_value=env,
    ) as load_mock, mock.patch.object(
        android_server, '_verify_prebaked_snapshots', return_value=True
    ), mock.patch.object(
        android_server.aw_registry_module, 'TaskRegistry',
        return_value=registry,
    ), mock.patch.object(
        android_server.suite_utils, 'create_suite', return_value=suite
    ), mock.patch.object(
        android_server, '_app_setup_list', return_value=()
    ):
      asyncio.run(android_server._initialize_async(app))

    self.assertTrue(app.state.ready)
    self.assertIs(app.state.app_android_env, env)
    self.assertIs(app.state.suite, suite)
    self.assertIs(app.state.task_registry, registry)
    self.assertIsNone(app.state.init_error)
    self.assertEqual(load_mock.call_args.kwargs['app_list'], ())

  def test_records_init_error_without_flipping_ready(self):
    app = mock.Mock()
    app.state = self._state()
    with mock.patch.object(
        android_server.asyncio, 'to_thread',
        new=mock.AsyncMock(side_effect=RuntimeError('boom')),
    ):
      asyncio.run(android_server._initialize_async(app))
    self.assertFalse(app.state.ready)
    self.assertIn('RuntimeError', app.state.init_error)
    self.assertIsNone(app.state.app_android_env)


class PrebakedSnapshotTest(absltest.TestCase):

  def _completed(self, returncode, stdout):
    import subprocess  # pylint: disable=g-import-not-at-top
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=''
    )

  def test_legacy_image_never_checks_snapshots(self):
    with mock.patch.object(
        android_server, '_PREBAKED', False
    ), mock.patch.object(
        android_server.subprocess, 'run'
    ) as run_mock:
      self.assertTrue(
          android_server._verify_prebaked_snapshots(mock.Mock())
      )
    run_mock.assert_not_called()

  def test_prebaked_nonempty_snapshot_dir_passes(self):
    with mock.patch.object(android_server, '_PREBAKED', True), \
        mock.patch.object(
            android_server.subprocess, 'run',
            return_value=self._completed(0, 'com.example.app\n'),
        ):
      self.assertTrue(
          android_server._verify_prebaked_snapshots(mock.Mock())
      )

  def test_prebaked_empty_snapshot_dir_triggers_fallback(self):
    with mock.patch.object(android_server, '_PREBAKED', True), \
        mock.patch.object(
            android_server.subprocess, 'run',
            return_value=self._completed(0, '\n'),
        ):
      self.assertFalse(
          android_server._verify_prebaked_snapshots(mock.Mock())
      )


class SoftRecoverySchedulingTest(absltest.TestCase):

  def test_non_adb_reason_never_starts_a_thread(self):
    with mock.patch.object(
        android_server, '_unhealthy_since', None
    ), mock.patch.object(
        android_server, 'threading'
    ) as threading_mock:
      self.assertFalse(
          android_server._request_soft_recovery('qemu_missing')
      )
    threading_mock.Thread.assert_not_called()

  def test_grace_period_blocks_immediate_recovery(self):
    with mock.patch.object(
        android_server, '_unhealthy_since', None
    ), mock.patch.object(
        android_server, '_RESTART_GRACE_SECONDS', 9999
    ), mock.patch.object(
        android_server, 'threading'
    ) as threading_mock:
      self.assertFalse(
          android_server._request_soft_recovery('adb_offline')
      )
    threading_mock.Thread.assert_not_called()


if __name__ == '__main__':
  absltest.main()
