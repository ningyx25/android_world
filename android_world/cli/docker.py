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

"""Thin subprocess wrappers around the ``docker`` CLI.

This is the lowest of the three ``aw env`` layers. Functions here only build
and execute docker commands; they contain no CLI parsing or presentation
logic. Everything goes through :func:`run_command` so callers can be tested by
patching a single function.
"""

import json
import logging
import os
import shlex
import subprocess

from android_world.cli import constants

__all__ = [
    'DockerCommandError',
    'run_command',
    'docker_ps',
    'docker_inspect',
    'docker_image_inspect',
    'list_managed_containers',
    'build_run_command',
    'docker_start',
    'docker_stop',
    'docker_rm',
    'docker_exec_replace',
    'docker_info',
    'docker_version',
]

logger = logging.getLogger('aw')


class DockerCommandError(RuntimeError):
  """Raised when a docker command exits with an unexpected status."""

  def __init__(self, cmd: list[str], returncode: int, stderr: str):
    self.cmd = cmd
    self.returncode = returncode
    self.stderr = stderr
    super().__init__(
        f"docker command failed ({returncode}): {shlex.join(cmd)}\n{stderr}"
    )


def run_command(
    cmd: list[str],
    capture: bool = True,
    allowed_exit_codes: set[int] | None = None,
) -> subprocess.CompletedProcess[str]:
  """Runs a command and returns the completed process.

  Args:
    cmd: Argument list, passed directly to ``subprocess.run`` (no shell).
    capture: Whether to capture stdout/stderr.
    allowed_exit_codes: Return codes considered successful. Defaults to {0}.

  Returns:
    The completed process.

  Raises:
    DockerCommandError: If the return code is not in ``allowed_exit_codes``.
  """
  acceptable = allowed_exit_codes if allowed_exit_codes is not None else {0}
  logger.debug('running: %s', shlex.join(cmd))
  result = subprocess.run(  # pylint: disable=subprocess-run-check
      cmd,
      capture_output=capture,
      text=True,
      check=False,
  )
  if result.returncode not in acceptable:
    stderr = result.stderr or ''
    if 'permission denied' in stderr.lower():
      stderr += (
          '\nHint: add your user to the docker group: '
          'sudo usermod -aG docker $USER && newgrp docker'
      )
    raise DockerCommandError(cmd, result.returncode, stderr)
  return result


def docker_ps(include_all: bool = False) -> list[dict]:
  """Returns raw ``docker ps`` rows as dicts.

  Args:
    include_all: Include stopped containers (``docker ps -a``).

  Returns:
    Parsed JSON rows. Malformed lines are skipped.
  """
  cmd = ['docker', 'ps', '--format', '{{json .}}']
  if include_all:
    cmd.insert(2, '-a')
  result = run_command(cmd)
  rows = []
  for line in result.stdout.splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError:
      logger.warning('skipping unparsable docker ps line: %r', line)
  return rows


def docker_inspect(name: str) -> dict | None:
  """Returns the inspect dict for ``name``, or None if it does not exist."""
  result = run_command(
      ['docker', 'inspect', name], allowed_exit_codes={0, 1}
  )
  if result.returncode != 0 or not result.stdout.strip():
    return None
  try:
    entries = json.loads(result.stdout)
  except json.JSONDecodeError:
    return None
  return entries[0] if entries else None


def docker_image_inspect(image: str) -> bool:
  """Returns True if the image exists locally (read-only)."""
  result = run_command(
      ['docker', 'image', 'inspect', image], allowed_exit_codes={0, 1}
  )
  return result.returncode == 0


def list_managed_containers(
    image_filter: str = constants.DEFAULT_IMAGE_FILTER,
    name_prefix: str = constants.DEFAULT_NAME_PREFIX,
    include_all: bool = False,
) -> list[dict]:
  """Returns docker ps rows belonging to the managed set.

  A container is managed only when **both** predicates hold: its name starts
  with ``name_prefix`` and its image contains ``image_filter`` (case
  insensitive). Two predicates are required because the host also runs
  containers with similar names (``screenjev-aw-*``) and similar images
  (``screenjev/androidworld:*``) that must never be touched.
  """
  image_needle = image_filter.lower()
  managed = []
  for row in docker_ps(include_all=include_all):
    name = row.get('Names', '') or ''
    image = row.get('Image', '') or ''
    if name.startswith(name_prefix) and image_needle in image.lower():
      managed.append(row)
  return managed


def build_run_command(
    *,
    name: str,
    image: str,
    host_port: int,
    container_port: int = constants.CONTAINER_PORT,
    detach: bool = True,
    restart_policy: str = constants.RESTART_POLICY,
    memory: str = constants.MEMORY,
    memory_swap: str = constants.MEMORY_SWAP,
) -> list[str]:
  """Builds the exact ``docker run`` argument list used by the shell scripts.

  Mirrors ``scripts/run_container.sh`` and ``server/restart_aw.sh`` flag for
  flag, including order.
  """
  cmd = [
      'docker',
      'run',
      '-d' if detach else '-it',
      '--init',
      '--privileged',
      '--name',
      name,
      '-p',
      f'{host_port}:{container_port}',
      '--restart',
      restart_policy,
      '--memory',
      memory,
      '--memory-swap',
      memory_swap,
      image,
  ]
  return cmd


def docker_start(name: str) -> subprocess.CompletedProcess[str]:
  """Starts a stopped container."""
  return run_command(['docker', 'start', name])


def docker_stop(name: str) -> subprocess.CompletedProcess[str]:
  """Stops a container."""
  return run_command(['docker', 'stop', name])


def docker_rm(
    name: str, force: bool = False
) -> subprocess.CompletedProcess[str]:
  """Removes a container."""
  cmd = ['docker', 'rm']
  if force:
    cmd.append('-f')
  cmd.append(name)
  return run_command(cmd)


def docker_exec_replace(
    name: str,
    command: str,
    interactive: bool = True,
) -> None:
  """Replaces the current process with ``docker exec``.

  Args:
    name: Container name.
    command: Command to run inside the container. Split with ``shlex``; pass a
      shell such as ``/bin/bash`` for an interactive shell.
    interactive: Whether to allocate a TTY (``-it``).
  """
  cmd = ['docker', 'exec']
  if interactive:
    cmd.append('-it')
  cmd.append(name)
  cmd.extend(shlex.split(command))
  logger.debug('replacing process with: %s', shlex.join(cmd))
  os.execvp('docker', cmd)


def docker_info() -> subprocess.CompletedProcess[str]:
  """Runs ``docker info`` (read-only daemon / permission probe)."""
  return run_command(['docker', 'info'], allowed_exit_codes={0, 1})


def docker_version() -> subprocess.CompletedProcess[str]:
  """Runs ``docker --version`` (read-only installation probe)."""
  return run_command(['docker', '--version'], allowed_exit_codes={0, 1})
