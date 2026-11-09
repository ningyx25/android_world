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

"""Tests for the bounded emulator health probes (no real adb is invoked)."""

import subprocess
from unittest import mock

from absl.testing import absltest

from server import health_probe


class HealthProbeTest(absltest.TestCase):

  def test_adb_device_state_classifies_output(self):
    cases = [
        ((0, 'device'), 'device'),
        ((1, 'offline'), 'offline'),
        ((1, 'unauthorized'), 'unauthorized'),
        ((1, 'error: no devices/emulators found'), 'missing'),
        ((None, ''), 'timeout'),
        ((0, 'weird'), 'error'),
    ]
    for adb_result, expected in cases:
      with self.subTest(expected=expected):
        with mock.patch.object(
            health_probe, '_adb', return_value=adb_result
        ):
          self.assertEqual(health_probe.adb_device_state(), expected)

  def test_adb_is_timeout_boxed(self):
    with mock.patch.object(
        health_probe.subprocess, 'run',
        side_effect=subprocess.TimeoutExpired(cmd=['adb'], timeout=3),
    ):
      self.assertEqual(health_probe._adb(['devices']), (None, ''))

  def test_adb_boot_completed_requires_exact_one(self):
    with mock.patch.object(health_probe, '_adb', return_value=(0, '1')):
      self.assertTrue(health_probe.adb_boot_completed())
    with mock.patch.object(health_probe, '_adb', return_value=(0, '')):
      self.assertFalse(health_probe.adb_boot_completed())
    with mock.patch.object(health_probe, '_adb', return_value=(0, '1\r')):
      self.assertTrue(health_probe.adb_boot_completed())

  def test_qemu_process_alive_from_pgrep(self):
    with mock.patch.object(
        health_probe.subprocess, 'run',
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout='', stderr=''
        ),
    ):
      self.assertTrue(health_probe.qemu_process_alive())

  def test_probe_device_qemu_missing_short_circuits(self):
    with mock.patch.object(
        health_probe, 'qemu_process_alive', return_value=False
    ), mock.patch.object(
        health_probe, 'adb_device_state'
    ) as state_mock:
      probe = health_probe.probe_device()
    self.assertFalse(probe['healthy'])
    self.assertEqual(probe['reason'], health_probe.REASON_QEMU_MISSING)
    state_mock.assert_not_called()

  def test_probe_device_adb_offline(self):
    with mock.patch.object(
        health_probe, 'qemu_process_alive', return_value=True
    ), mock.patch.object(
        health_probe, 'adb_device_state', return_value='offline'
    ):
      probe = health_probe.probe_device()
    self.assertFalse(probe['healthy'])
    self.assertEqual(probe['reason'], health_probe.REASON_ADB_OFFLINE)

  def test_probe_device_boot_not_completed(self):
    with mock.patch.object(
        health_probe, 'qemu_process_alive', return_value=True
    ), mock.patch.object(
        health_probe, 'adb_device_state', return_value='device'
    ), mock.patch.object(
        health_probe, 'adb_boot_completed', return_value=False
    ):
      probe = health_probe.probe_device()
    self.assertFalse(probe['healthy'])
    self.assertEqual(
        probe['reason'], health_probe.REASON_BOOT_NOT_COMPLETED
    )

  def test_probe_device_healthy(self):
    with mock.patch.object(
        health_probe, 'qemu_process_alive', return_value=True
    ), mock.patch.object(
        health_probe, 'adb_device_state', return_value='device'
    ), mock.patch.object(
        health_probe, 'adb_boot_completed', return_value=True
    ):
      probe = health_probe.probe_device()
    self.assertTrue(probe['healthy'])
    self.assertEqual(probe['reason'], health_probe.REASON_OK)

  def test_wait_for_boot_returns_immediately_when_healthy(self):
    with mock.patch.object(
        health_probe, 'probe_device', return_value={'healthy': True}
    ), mock.patch.object(health_probe.time, 'sleep') as sleep_mock:
      self.assertTrue(health_probe.wait_for_boot(timeout=10, interval=1))
    sleep_mock.assert_not_called()

  def test_wait_for_boot_times_out(self):
    with mock.patch.object(
        health_probe, 'probe_device', return_value={'healthy': False}
    ):
      self.assertFalse(health_probe.wait_for_boot(timeout=0))

  def test_soft_recovery_never_touches_qemu(self):
    with mock.patch.object(
        health_probe, 'adb_reconnect'
    ) as reconnect_mock, mock.patch.object(
        health_probe, 'wait_for_device', return_value=True
    ), mock.patch.object(
        health_probe, 'wait_for_boot', return_value=True
    ), mock.patch.object(
        health_probe, 'qemu_process_alive'
    ) as qemu_mock:
      self.assertTrue(health_probe.soft_recovery(timeout=20))
    reconnect_mock.assert_called_once()
    qemu_mock.assert_not_called()


if __name__ == '__main__':
  absltest.main()
