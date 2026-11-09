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

"""Tests for the docker CLI wrappers (no real docker is invoked)."""

import json
import subprocess
from unittest import mock

from absl.testing import absltest

from android_world.cli import docker


def _completed(returncode=0, stdout='', stderr=''):
  return subprocess.CompletedProcess(
      args=[], returncode=returncode, stdout=stdout, stderr=stderr
  )


class DockerTest(absltest.TestCase):

  def test_build_run_command_matches_shell_script_exactly(self):
    self.assertEqual(
        docker.build_run_command(
            name='aw_3800', image='android_world:latest', host_port=3800
        ),
        [
            'docker', 'run', '-d', '--init', '--privileged',
            '--name', 'aw_3800', '-p', '3800:5000',
            '--restart', 'unless-stopped',
            '--memory', '6g', '--memory-swap', '6g',
            'android_world:latest',
        ],
    )

  def test_build_run_command_non_detached_uses_it(self):
    cmd = docker.build_run_command(
        name='aw_3801', image='android_world:latest', host_port=3801,
        detach=False,
    )
    self.assertEqual(cmd[2], '-it')
    self.assertNotIn('-d', cmd)

  def test_build_run_command_honours_overrides(self):
    cmd = docker.build_run_command(
        name='x', image='img:v2', host_port=9999, container_port=5000,
        restart_policy='no', memory='8g', memory_swap='8g',
    )
    self.assertIn('--restart', cmd)
    self.assertEqual(cmd[cmd.index('--restart') + 1], 'no')
    self.assertEqual(cmd[cmd.index('--memory') + 1], '8g')
    self.assertEqual(cmd[-1], 'img:v2')

  def test_docker_ps_parses_json_lines_and_all_flag(self):
    rows = [
        {'Names': 'aw_15000', 'Image': 'android_world:latest',
         'State': 'running'},
        {'Names': 'foo', 'Image': 'bar', 'State': 'running'},
    ]
    stdout = '\n'.join(json.dumps(row) for row in rows) + '\n'
    with mock.patch.object(
        docker, 'run_command', return_value=_completed(stdout=stdout)
    ) as run_mock:
      self.assertEqual(docker.docker_ps(), rows)
      self.assertEqual(
          run_mock.call_args[0][0],
          ['docker', 'ps', '--format', '{{json .}}'],
      )
    with mock.patch.object(
        docker, 'run_command', return_value=_completed(stdout='')
    ) as run_mock:
      docker.docker_ps(include_all=True)
      self.assertEqual(
          run_mock.call_args[0][0],
          ['docker', 'ps', '-a', '--format', '{{json .}}'],
      )

  def test_docker_ps_skips_malformed_lines(self):
    with mock.patch.object(
        docker, 'run_command',
        return_value=_completed(stdout='not json\n\n{"Names":"aw_1"}\n'),
    ):
      self.assertEqual(docker.docker_ps(), [{'Names': 'aw_1'}])

  def test_docker_inspect_returns_none_when_missing(self):
    with mock.patch.object(
        docker, 'run_command', return_value=_completed(stdout='[]\n')
    ):
      self.assertIsNone(docker.docker_inspect('aw_missing'))
    with mock.patch.object(
        docker, 'run_command', return_value=_completed(returncode=1)
    ):
      self.assertIsNone(docker.docker_inspect('aw_missing'))

  def test_docker_inspect_returns_first_entry(self):
    payload = [{'Name': '/aw_15000', 'State': {'Running': True}}]
    with mock.patch.object(
        docker, 'run_command',
        return_value=_completed(stdout=json.dumps(payload)),
    ):
      self.assertEqual(docker.docker_inspect('aw_15000'), payload[0])

  def test_docker_image_inspect_is_read_only_boolean(self):
    with mock.patch.object(
        docker, 'run_command', return_value=_completed(returncode=0)
    ) as run_mock:
      self.assertTrue(docker.docker_image_inspect('android_world:latest'))
      self.assertEqual(
          run_mock.call_args[0][0],
          ['docker', 'image', 'inspect', 'android_world:latest'],
      )
    with mock.patch.object(
        docker, 'run_command', return_value=_completed(returncode=1)
    ):
      self.assertFalse(docker.docker_image_inspect('missing:latest'))

  def test_list_managed_containers_requires_both_predicates(self):
    rows = [
        {'Names': 'aw_15000', 'Image': 'android_world:latest'},
        {'Names': 'aw_15001', 'Image': 'android_world:v2'},
        {'Names': 'screenjev-aw-r3-069', 'Image': 'android_world:latest'},
        {'Names': 'screenjev-aw-r3-069',
         'Image': 'screenjev/androidworld:20261029'},
        {'Names': 'mobile_world_env_0',
         'Image': 'ghcr.io/tongyi-mai/mobile_world:latest'},
        {'Names': 'aw_9999', 'Image': 'some/other:latest'},
    ]
    with mock.patch.object(docker, 'docker_ps', return_value=rows):
      managed = docker.list_managed_containers()
    self.assertEqual(
        [row['Names'] for row in managed], ['aw_15000', 'aw_15001']
    )

  def test_run_command_raises_on_unexpected_exit_code(self):
    with mock.patch.object(
        docker.subprocess, 'run',
        return_value=_completed(returncode=1, stderr='boom'),
    ):
      with self.assertRaises(docker.DockerCommandError):
        docker.run_command(['docker', 'ps'])

  def test_run_command_allows_declared_exit_codes(self):
    with mock.patch.object(
        docker.subprocess, 'run',
        return_value=_completed(returncode=1, stderr=''),
    ):
      result = docker.run_command(
          ['docker', 'inspect', 'x'], allowed_exit_codes={0, 1}
      )
      self.assertEqual(result.returncode, 1)

  def test_docker_rm_adds_force_flag(self):
    with mock.patch.object(
        docker, 'run_command', return_value=_completed()
    ) as run_mock:
      docker.docker_rm('aw_3800', force=True)
      self.assertEqual(
          run_mock.call_args[0][0], ['docker', 'rm', '-f', 'aw_3800']
      )

  def test_docker_exec_replace_uses_execvp(self):
    with mock.patch.object(docker.os, 'execvp') as exec_mock:
      docker.docker_exec_replace('aw_3800', '/bin/bash')
    self.assertEqual(
        exec_mock.call_args[0][1],
        ['docker', 'exec', '-it', 'aw_3800', '/bin/bash'],
    )


if __name__ == '__main__':
  absltest.main()
