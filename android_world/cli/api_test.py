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

"""Tests for the programmatic environment API (all docker calls mocked)."""

import subprocess
from unittest import mock

from absl.testing import absltest
import requests

from android_world.cli import api
from android_world.cli import constants
from android_world.cli import docker


def _completed(returncode=0, stdout='', stderr=''):
  return subprocess.CompletedProcess(
      args=[], returncode=returncode, stdout=stdout, stderr=stderr
  )


_EXPECTED_RUN = [
    'docker', 'run', '-d', '--init', '--privileged',
    '--name', 'aw_3800', '-p', '3800:5000',
    '--restart', 'unless-stopped',
    '--memory', '6g', '--memory-swap', '6g',
    'android_world:latest',
]


class ApiTest(absltest.TestCase):

  def test_resolve_name_expands_numeric_port(self):
    self.assertEqual(api.resolve_name('15007'), 'aw_15007')
    self.assertEqual(api.resolve_name('aw_15007'), 'aw_15007')
    self.assertEqual(api.resolve_name('/aw_15007'), 'aw_15007')
    self.assertEqual(api.resolve_name('3800', prefix='zz_'), 'zz_3800')

  def test_port_helpers(self):
    self.assertEqual(
        api._parse_host_port('0.0.0.0:15007->5000/tcp, :::15007->5000/tcp'),
        15007,
    )
    self.assertIsNone(api._parse_host_port(''))
    self.assertEqual(api._port_from_name('aw_15007', 'aw_'), 15007)
    self.assertIsNone(api._port_from_name('screenjev-aw-1', 'aw_'))

  def test_find_available_ports_skips_used_and_unbindable(self):
    with mock.patch.object(api, 'managed_ports', return_value={3800}), \
         mock.patch.object(
             api, 'is_port_available',
             side_effect=lambda port: port != 3801,
         ):
      self.assertEqual(
          api.find_available_ports(2, start_port=3800), [3802, 3803]
      )

  def test_get_container_info_parses_inspect(self):
    raw = {
        'Name': '/aw_15000',
        'Config': {'Image': 'android_world:latest'},
        'State': {'Status': 'running', 'Running': True,
                  'StartedAt': '2026-11-03T13:48:37Z'},
        'Created': '2026-11-03T13:48:36Z',
        'HostConfig': {'Memory': 6 * 1024 ** 3,
                       'RestartPolicy': {'Name': 'unless-stopped'}},
        'NetworkSettings': {'Ports': {'5000/tcp': [{'HostPort': '15000'}]}},
    }
    with mock.patch.object(api.docker, 'docker_inspect', return_value=raw):
      info = api.get_container_info('aw_15000')
    self.assertIsNotNone(info)
    self.assertEqual(info.name, 'aw_15000')
    self.assertEqual(info.host_port, 15000)
    self.assertEqual(info.restart_policy, 'unless-stopped')
    self.assertEqual(info.memory, '6g')
    self.assertTrue(info.running)

  def test_get_container_info_falls_back_to_name_port(self):
    raw = {'Name': '/aw_3800', 'State': {}, 'Config': {},
           'NetworkSettings': {'Ports': {}}}
    with mock.patch.object(api.docker, 'docker_inspect', return_value=raw):
      info = api.get_container_info('aw_3800')
    self.assertEqual(info.host_port, 3800)

  def test_list_containers_maps_docker_ps_rows(self):
    rows = [{
        'Names': 'aw_15007', 'Image': 'android_world:latest',
        'Status': 'Up 22 minutes', 'State': 'running',
        'Ports': '0.0.0.0:15007->5000/tcp',
        'CreatedAt': '2026-11-09 21:07:20 +0800 CST',
    }]
    with mock.patch.object(
        api.docker, 'list_managed_containers', return_value=rows
    ) as list_mock:
      infos = api.list_containers(include_all=True)
    self.assertEqual(list_mock.call_args[0][2], True)
    self.assertEqual(infos[0].host_port, 15007)
    self.assertTrue(infos[0].running)

  def test_probe_health(self):
    with mock.patch.object(
        api.requests, 'get', return_value=mock.Mock(status_code=200)
    ):
      self.assertTrue(api.probe_health(3800))
    with mock.patch.object(
        api.requests, 'get', return_value=mock.Mock(status_code=503)
    ):
      self.assertFalse(api.probe_health(3800))
    with mock.patch.object(
        api.requests, 'get', side_effect=requests.RequestException('down')
    ):
      self.assertFalse(api.probe_health(3800))

  def test_launch_dry_run_makes_zero_docker_calls(self):
    with mock.patch.object(
        api.docker, 'docker_image_inspect'
    ) as image_mock, \
        mock.patch.object(api.docker, 'docker_inspect') as inspect_mock, \
        mock.patch.object(api.docker, 'run_command') as run_mock:
      results = api.launch_containers([3800], dry_run=True)
    image_mock.assert_not_called()
    inspect_mock.assert_not_called()
    run_mock.assert_not_called()
    self.assertEqual(results[0].status, 'dry_run')
    self.assertEqual(results[0].command, _EXPECTED_RUN)

  def test_launch_skips_when_image_missing(self):
    with mock.patch.object(
        api.docker, 'docker_image_inspect', return_value=False
    ), \
        mock.patch.object(api.docker, 'docker_inspect') as inspect_mock, \
        mock.patch.object(api.docker, 'run_command') as run_mock:
      results = api.launch_containers([3800])
    self.assertEqual(results[0].status, 'image_missing')
    self.assertIn('docker build -t android_world:latest .',
                  results[0].message)
    inspect_mock.assert_not_called()
    run_mock.assert_not_called()

  def test_launch_skips_existing_container_without_mutating(self):
    with mock.patch.object(
        api.docker, 'docker_image_inspect', return_value=True
    ), \
        mock.patch.object(
            api.docker, 'docker_inspect', return_value={'Name': '/aw_3800'}
        ), \
        mock.patch.object(api.docker, 'run_command') as run_mock, \
        mock.patch.object(api.docker, 'docker_rm') as rm_mock, \
        mock.patch.object(api.docker, 'docker_stop') as stop_mock:
      results = api.launch_containers([3800])
    self.assertEqual(results[0].status, 'already_exists')
    run_mock.assert_not_called()
    rm_mock.assert_not_called()
    stop_mock.assert_not_called()

  def test_launch_runs_exact_docker_command(self):
    with mock.patch.object(
        api.docker, 'docker_image_inspect', return_value=True
    ), \
        mock.patch.object(api.docker, 'docker_inspect', return_value=None), \
        mock.patch.object(
            api.docker, 'run_command', return_value=_completed()
        ) as run_mock:
      results = api.launch_containers([3800])
    self.assertEqual(results[0].status, 'launched')
    self.assertEqual(run_mock.call_args[0][0], _EXPECTED_RUN)

  def test_launch_wait_uses_health_poll(self):
    with mock.patch.object(
        api.docker, 'docker_image_inspect', return_value=True
    ), \
        mock.patch.object(api.docker, 'docker_inspect', return_value=None), \
        mock.patch.object(api.docker, 'run_command', return_value=_completed()), \
        mock.patch.object(api, 'wait_for_healthy', return_value=True) as wait_mock:
      results = api.launch_containers([3800], wait=True)
    wait_mock.assert_called_once_with(3800)
    self.assertEqual(results[0].message, 'healthy')

  def test_remove_containers_only_touches_managed_set(self):
    rows = [
        {'Names': 'aw_15000'}, {'Names': 'aw_15001'},
    ]
    with mock.patch.object(
        api.docker, 'list_managed_containers', return_value=rows
    ), \
        mock.patch.object(api.docker, 'docker_rm') as rm_mock:
      removed, skipped = api.remove_containers(['9999', '15000'])
    self.assertEqual(removed, ['aw_15000'])
    self.assertTrue(any('aw_9999' in entry for entry in skipped))
    rm_mock.assert_called_once_with('aw_15000', force=False)

  def test_remove_all_uses_force_when_requested(self):
    rows = [{'Names': 'aw_15000'}, {'Names': 'aw_15001'}]
    with mock.patch.object(
        api.docker, 'list_managed_containers', return_value=rows
    ), \
        mock.patch.object(api.docker, 'docker_rm') as rm_mock:
      removed, skipped = api.remove_containers(
          names=None, all_=True, force=True
      )
    self.assertEqual(removed, ['aw_15000', 'aw_15001'])
    self.assertEqual(skipped, [])
    self.assertTrue(all(call[1]['force'] for call in rm_mock.call_args_list))

  def test_restart_container_command_sequence(self):
    info = constants.ContainerInfo(
        name='aw_15000', image='android_world:latest', host_port=15000
    )
    with mock.patch.object(api, 'get_container_info', return_value=info), \
        mock.patch.object(api.docker, 'docker_stop') as stop_mock, \
        mock.patch.object(api.docker, 'docker_rm') as rm_mock, \
        mock.patch.object(api.docker, 'run_command') as run_mock:
      parent = mock.Mock()
      parent.attach_mock(stop_mock, 'stop')
      parent.attach_mock(rm_mock, 'rm')
      parent.attach_mock(run_mock, 'run')
      cmd = api.restart_container('aw_15000')
    self.assertEqual(
        cmd,
        [
            'docker', 'run', '-d', '--init', '--privileged',
            '--name', 'aw_15000', '-p', '15000:5000',
            '--restart', 'unless-stopped',
            '--memory', '6g', '--memory-swap', '6g',
            'android_world:latest',
        ],
    )
    self.assertEqual(
        [call[0] for call in parent.mock_calls], ['stop', 'rm', 'run']
    )

  def test_restart_requires_a_determinable_port(self):
    with mock.patch.object(api, 'get_container_info', return_value=None):
      with self.assertRaises(ValueError):
        api.restart_container('weird_name')

  def test_check_prerequisites_missing_image_never_builds(self):
    with mock.patch.object(api.shutil, 'which', return_value='/usr/bin/docker'), \
        mock.patch.object(
            api.docker, 'docker_version',
            return_value=_completed(stdout='Docker version 24.0.7'),
        ), \
        mock.patch.object(
            api.docker, 'docker_info', return_value=_completed(),
        ), \
        mock.patch.object(
            api.docker, 'docker_image_inspect', return_value=False
        ), \
        mock.patch.object(api.docker, 'run_command') as run_mock:
      results = api.check_prerequisites()
    image_result = [r for r in results if r.name.startswith('Image ')][0]
    self.assertFalse(image_result.passed)
    self.assertIn('docker build -t android_world:latest .', image_result.hint)
    run_mock.assert_not_called()

  def test_check_prerequisites_short_circuits_without_docker(self):
    with mock.patch.object(api.shutil, 'which', return_value=None), \
        mock.patch.object(api.docker, 'run_command') as run_mock:
      results = api.check_prerequisites()
    self.assertEqual(len(results), 1)
    self.assertFalse(results[0].passed)
    run_mock.assert_not_called()


class RestartServerInPlaceTest(absltest.TestCase):

  def test_kills_server_and_polls_health(self):
    with mock.patch.object(
        api, 'get_container_info',
        return_value=constants.ContainerInfo(host_port=3800),
    ), mock.patch.object(
        api.docker, 'run_command'
    ) as run_mock, mock.patch.object(
        api, 'probe_health', side_effect=[False, True]
    ) as health_mock, mock.patch.object(api.time, 'sleep'):
      ok = api.restart_server_in_container('aw_3800')
    self.assertTrue(ok)
    self.assertEqual(
        run_mock.call_args[0][0],
        ['docker', 'exec', 'aw_3800', 'pkill', '-f',
         'server[.]android_server'],
    )
    self.assertEqual(health_mock.call_count, 2)

  def test_uses_exact_docker_exec_command(self):
    with mock.patch.object(
        api, 'get_container_info',
        return_value=constants.ContainerInfo(host_port=3800),
    ), mock.patch.object(api.docker, 'run_command') as run_mock, \
        mock.patch.object(api, 'probe_health', return_value=True), \
        mock.patch.object(api.time, 'sleep'):
      api.restart_server_in_container('aw_3800')
    self.assertEqual(
        run_mock.call_args.kwargs['allowed_exit_codes'], {0, 1, 137, 143}
    )

  def test_unknown_port_raises(self):
    with mock.patch.object(api, 'get_container_info', return_value=None):
      with self.assertRaises(ValueError):
        api.restart_server_in_container('weird_name')


if __name__ == '__main__':
  absltest.main()
