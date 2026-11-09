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

"""Tests for argparse registration, dispatch and rich presentation."""

import argparse
import io
import json
import logging
import shlex
from unittest import mock

from absl.testing import absltest

from android_world.cli import api
from android_world.cli import constants
from android_world.cli import docker
from android_world.cli import env
from android_world.cli import main

_EXPECTED_RUN = [
    'docker', 'run', '-d', '--init', '--privileged',
    '--name', 'aw_3800', '-p', '3800:5000',
    '--restart', 'unless-stopped',
    '--memory', '6g', '--memory-swap', '6g',
    'android_world:latest',
]


class _FakeTable:
  """Minimal stand-in for rich.table.Table that records rows."""

  def __init__(self, *args, **kwargs):
    del args, kwargs
    self.columns = []
    self.rows = []

  def add_column(self, *args, **kwargs):
    del kwargs
    self.columns.append(args)

  def add_row(self, *cells):
    self.rows.append(cells)


class EnvTest(absltest.TestCase):

  def _parse(self, argv):
    return main.create_parser().parse_args(argv)

  def test_run_defaults(self):
    args = self._parse(['env', 'run'])
    self.assertEqual(args.command, 'env')
    self.assertEqual(args.env_action, 'run')
    self.assertEqual(args.port, constants.DEFAULT_HOST_PORT)
    self.assertEqual(args.port, 3800)
    self.assertEqual(args.count, 1)
    self.assertEqual(args.image, constants.DEFAULT_IMAGE)
    self.assertFalse(args.dry_run)
    self.assertFalse(args.wait)
    self.assertTrue(args.detach)

  def test_run_flags_and_dry_run(self):
    args = self._parse([
        'env', 'run', '--port', '3900', '--count', '3', '--dry-run',
        '--wait', '--no-detach', '--image', 'custom:v2',
    ])
    self.assertEqual(args.port, 3900)
    self.assertEqual(args.count, 3)
    self.assertTrue(args.dry_run)
    self.assertTrue(args.wait)
    self.assertFalse(args.detach)
    self.assertEqual(args.image, 'custom:v2')

  def test_common_flags_have_hyphen_and_underscore_forms(self):
    self.assertEqual(
        self._parse(['env', 'run', '--name_prefix', 'zz_']).name_prefix,
        'zz_',
    )
    self.assertEqual(
        self._parse(['env', 'run', '--prefix', 'yy_']).name_prefix,
        'yy_',
    )
    self.assertEqual(
        self._parse(['env', 'run', '--name-prefix', 'xx_']).name_prefix,
        'xx_',
    )

  def test_quiet_and_verbose_flags(self):
    args = self._parse(['env', '-q', 'list'])
    self.assertTrue(args.quiet)
    self.assertEqual(args.verbose, 0)
    args = self._parse(['env', '-v', '-v', 'list'])
    self.assertEqual(args.verbose, 2)
    self.assertFalse(args.quiet)

  def test_verbosity_maps_to_logging_level(self):
    with mock.patch.object(logging, 'basicConfig'):
      self.assertEqual(
          main._set_log_level(self._parse(['env', '-q', 'list'])),
          logging.WARNING,
      )
      self.assertEqual(
          main._set_log_level(self._parse(['env', '-v', 'list'])),
          logging.DEBUG,
      )
      self.assertEqual(
          main._set_log_level(self._parse(['env', '-vv', 'list'])),
          logging.DEBUG,
      )

  def test_list_alias_and_flags(self):
    args = self._parse(['env', 'ls', '--all', '--json', '--health'])
    self.assertEqual(args.env_action, 'ls')
    self.assertTrue(args.all_)
    self.assertTrue(args.as_json)
    self.assertTrue(args.health)

  def test_exec_command_default_and_override(self):
    self.assertEqual(
        self._parse(['env', 'exec', 'aw_1']).command, '/bin/bash'
    )
    self.assertEqual(
        self._parse(['env', 'exec', 'aw_1', '-c', 'ls -la']).command,
        'ls -la',
    )

  def test_all_subcommands_are_registered(self):
    cases = [
        (['env', 'run'], 'run'),
        (['env', 'rm'], 'rm'),
        (['env', 'list'], 'list'),
        (['env', 'ls'], 'ls'),
        (['env', 'info', 'aw_15000'], 'info'),
        (['env', 'restart', 'aw_15000'], 'restart'),
        (['env', 'exec', 'aw_15000'], 'exec'),
        (['env', 'check'], 'check'),
    ]
    for argv, action in cases:
      self.assertEqual(self._parse(argv).env_action, action)

  def test_execute_dispatches_every_action(self):
    cases = [
        (['env', 'run'], '_run'),
        (['env', 'rm'], '_rm'),
        (['env', 'list'], '_list'),
        (['env', 'ls'], '_list'),
        (['env', 'info', 'aw_15000'], '_info'),
        (['env', 'restart', 'aw_15000'], '_restart'),
        (['env', 'exec', 'aw_15000'], '_exec'),
        (['env', 'check'], '_check'),
    ]
    for argv, handler_name in cases:
      args = self._parse(argv)
      with mock.patch.object(env, handler_name) as handler:
        env.execute(args)
        handler.assert_called_once_with(args)

  def test_execute_rejects_unknown_action(self):
    args = argparse.Namespace(env_action='bogus')
    with self.assertRaises(ValueError):
      env.execute(args)

  def test_dry_run_prints_exact_command_and_runs_nothing(self):
    created = []

    def make_table(*args, **kwargs):
      del args, kwargs
      table = _FakeTable()
      created.append(table)
      return table

    args = self._parse(['env', 'run', '--dry-run'])
    with mock.patch.object(api, 'find_available_ports', return_value=[3800]), \
        mock.patch.object(env, 'Table', side_effect=make_table), \
        mock.patch.object(env.console, 'print'), \
        mock.patch.object(docker, 'run_command') as run_mock, \
        mock.patch.object(docker, 'docker_inspect') as inspect_mock, \
        mock.patch.object(docker, 'docker_image_inspect') as image_mock:
      env.execute(args)
    run_mock.assert_not_called()
    inspect_mock.assert_not_called()
    image_mock.assert_not_called()
    self.assertEqual(created[0].rows[0][2], shlex.join(_EXPECTED_RUN))

  def test_list_json_is_valid_json(self):
    info = constants.ContainerInfo(
        name='aw_15000', image='android_world:latest',
        status='Up 6 days', running=True, host_port=15000,
    )
    args = self._parse(['env', 'list', '--json'])
    buffer = io.StringIO()
    with mock.patch.object(api, 'list_containers', return_value=[info]), \
        mock.patch('sys.stdout', buffer):
      env.execute(args)
    payload = json.loads(buffer.getvalue())
    self.assertEqual(payload[0]['name'], 'aw_15000')
    self.assertEqual(payload[0]['host_port'], 15000)
    self.assertIsNone(payload[0]['health'])

  def test_check_json_is_valid_json(self):
    results = [constants.CheckResult(name='Docker', passed=True,
                                    message='ok')]
    args = self._parse(['env', 'check', '--json'])
    buffer = io.StringIO()
    with mock.patch.object(api, 'check_prerequisites', return_value=results), \
        mock.patch('sys.stdout', buffer):
      env.execute(args)
    payload = json.loads(buffer.getvalue())
    self.assertEqual(payload[0]['name'], 'Docker')
    self.assertTrue(payload[0]['passed'])


if __name__ == '__main__':
  absltest.main()
