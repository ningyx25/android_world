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

"""Programmatic Docker environment API for the ``aw env`` CLI.

This middle layer contains the environment logic (port allocation, container
selection, health probing, prerequisite checks). It is intentionally free of
argparse and rich so it can be reused and unit tested in isolation. All docker
access goes through :mod:`android_world.cli.docker`.
"""

import logging
import os
import re
import shutil
import socket
import time

import requests

from android_world.cli import constants
from android_world.cli import docker

logger = logging.getLogger('aw')


def resolve_name(
    name_or_port: str, prefix: str = constants.DEFAULT_NAME_PREFIX
) -> str:
  """Expands a numeric host-port shorthand to a container name.

  Args:
    name_or_port: Either a full container name or a numeric host port.
    prefix: Name prefix, ``aw_`` by default.

  Returns:
    The container name, e.g. ``"15007"`` becomes ``"aw_15007"``.
  """
  text = str(name_or_port).strip()
  if text.isdigit():
    return f'{prefix}{text}'
  return text.lstrip('/')


def _row_name(row: dict) -> str:
  """Returns the primary container name from a ``docker ps`` JSON row."""
  names = row.get('Names') or ''
  return names.split(',')[0].strip().lstrip('/')


def _port_from_name(name: str, prefix: str) -> int | None:
  """Extracts the host port from a ``<prefix><port>`` container name."""
  match = re.fullmatch(rf'{re.escape(prefix)}(\d+)', name)
  return int(match.group(1)) if match else None


def _parse_host_port(ports: str) -> int | None:
  """Parses a published host port out of a ``docker ps`` Ports string."""
  match = re.search(r'(\d+)->\d+/tcp', ports or '')
  return int(match.group(1)) if match else None


def list_containers(
    image_filter: str = constants.DEFAULT_IMAGE_FILTER,
    name_prefix: str = constants.DEFAULT_NAME_PREFIX,
    include_all: bool = False,
) -> list[constants.ContainerInfo]:
  """Lists managed containers as :class:`ContainerInfo` objects."""
  infos = []
  for row in docker.list_managed_containers(
      image_filter, name_prefix, include_all
  ):
    name = _row_name(row)
    status = row.get('Status') or ''
    host_port = _parse_host_port(row.get('Ports') or '')
    if host_port is None:
      host_port = _port_from_name(name, name_prefix)
    infos.append(
        constants.ContainerInfo(
            name=name,
            image=row.get('Image') or '',
            status=status,
            running=(row.get('State') == 'running') or 'Up' in status,
            host_port=host_port,
            created_at=row.get('CreatedAt') or '',
            state=row.get('State') or '',
        )
    )
  return infos


def get_container_info(
    name: str, name_prefix: str = constants.DEFAULT_NAME_PREFIX
) -> constants.ContainerInfo | None:
  """Returns detailed info from ``docker inspect``, or None if missing."""
  raw = docker.docker_inspect(name)
  if raw is None:
    return None

  actual_name = (raw.get('Name') or name).lstrip('/')
  state = raw.get('State') or {}
  config = raw.get('Config') or {}
  host_config = raw.get('HostConfig') or {}
  network_ports = (raw.get('NetworkSettings') or {}).get('Ports') or {}

  host_port = None
  bindings = network_ports.get(f'{constants.CONTAINER_PORT}/tcp') or []
  if bindings:
    try:
      host_port = int(bindings[0].get('HostPort'))
    except (TypeError, ValueError):
      host_port = None
  if host_port is None:
    host_port = _port_from_name(actual_name, name_prefix)

  memory_bytes = host_config.get('Memory') or 0
  memory = f'{memory_bytes // (1024 ** 3)}g' if memory_bytes else ''
  restart_policy = (host_config.get('RestartPolicy') or {}).get('Name') or ''

  return constants.ContainerInfo(
      name=actual_name,
      image=config.get('Image') or '',
      status=state.get('Status') or '',
      running=bool(state.get('Running')),
      host_port=host_port,
      created_at=raw.get('Created') or '',
      state=state.get('Status') or '',
      started_at=state.get('StartedAt') or '',
      restart_policy=restart_policy,
      memory=memory,
  )


def is_port_available(port: int) -> bool:
  """Returns True if ``port`` can be bound on ``0.0.0.0`` right now."""
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
      sock.bind(('0.0.0.0', port))
      return True
    except OSError:
      return False


def managed_ports(
    image_filter: str = constants.DEFAULT_IMAGE_FILTER,
    name_prefix: str = constants.DEFAULT_NAME_PREFIX,
) -> set[int]:
  """Returns host ports already used by managed (including stopped) containers."""
  ports = set()
  for row in docker.list_managed_containers(
      image_filter, name_prefix, include_all=True
  ):
    port = _port_from_name(_row_name(row), name_prefix)
    if port is not None:
      ports.add(port)
  return ports


def find_available_ports(
    count: int,
    start_port: int = constants.DEFAULT_HOST_PORT,
    existing_ports: set[int] | None = None,
) -> list[int]:
  """Finds ``count`` free host ports starting at ``start_port``.

  Skips ports already claimed by a managed container and ports that cannot be
  bound on the host.
  """
  used = set(existing_ports) if existing_ports is not None else managed_ports()
  found = []
  port = start_port
  while len(found) < count and port <= 65535:
    if port not in used and is_port_available(port):
      found.append(port)
      used.add(port)
    port += 1
  return found


def probe_health(
    port: int, timeout: float = constants.HEALTH_PROBE_TIMEOUT_SEC
) -> bool:
  """Returns True if ``GET /health`` on ``port`` answers with a 2xx status."""
  url = f'http://localhost:{port}{constants.HEALTH_PATH}'
  try:
    response = requests.get(url, timeout=timeout)
    return 200 <= response.status_code < 300
  except requests.RequestException:
    return False


def wait_for_healthy(
    port: int,
    timeout: float = constants.HEALTH_TIMEOUT_SEC,
    poll: float = constants.HEALTH_POLL_SEC,
) -> bool:
  """Polls ``/health`` until it returns 2xx or ``timeout`` elapses."""
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if probe_health(port):
      return True
    time.sleep(poll)
  return probe_health(port)


def launch_containers(
    ports: list[int],
    *,
    image: str = constants.DEFAULT_IMAGE,
    name_prefix: str = constants.DEFAULT_NAME_PREFIX,
    detach: bool = True,
    dry_run: bool = False,
    wait: bool = False,
) -> list[constants.LaunchResult]:
  """Launches one container per requested host port.

  ``dry_run`` performs **no** docker subprocess calls at all: it only builds
  and returns the exact ``docker run`` commands. For a real launch, a port is
  skipped when a container with that name already exists (any state) or when
  the image is missing locally; nothing is ever stopped, removed or rebuilt.
  """
  results = []
  if not dry_run and not docker.docker_image_inspect(image):
    hint = f'image {image} not found; build it with: docker build -t {image} .'
    return [
        constants.LaunchResult(
            name=f'{name_prefix}{port}', port=port,
            status='image_missing', message=hint,
        )
        for port in ports
    ]

  for port in ports:
    name = f'{name_prefix}{port}'
    cmd = docker.build_run_command(
        name=name, image=image, host_port=port, detach=detach
    )
    if dry_run:
      results.append(
          constants.LaunchResult(
              name=name, port=port, status='dry_run', command=cmd,
              message='dry-run: no container created',
          )
      )
      continue
    if docker.docker_inspect(name) is not None:
      results.append(
          constants.LaunchResult(
              name=name, port=port, status='already_exists',
              message=f'{name} already exists; left untouched',
          )
      )
      continue
    try:
      docker.run_command(cmd)
    except docker.DockerCommandError as exc:
      results.append(
          constants.LaunchResult(
              name=name, port=port, status='failed', command=cmd,
              message=str(exc),
          )
      )
      continue
    if wait:
      healthy = wait_for_healthy(port)
      message = 'healthy' if healthy else 'launched (health check timed out)'
    else:
      message = 'launched'
    results.append(
        constants.LaunchResult(
            name=name, port=port, status='launched', command=cmd,
            message=message,
        )
    )
  return results


def remove_containers(
    names: list[str] | None = None,
    *,
    all_: bool = False,
    force: bool = False,
    image_filter: str = constants.DEFAULT_IMAGE_FILTER,
    name_prefix: str = constants.DEFAULT_NAME_PREFIX,
) -> tuple[list[str], list[str]]:
  """Removes managed containers, returning ``(removed, skipped)``.

  Only containers in the double-filtered managed set are ever targeted. With
  no explicit names (or with ``all_``) every managed container is selected,
  matching the reference ``mw env rm`` behaviour.
  """
  managed = {
      _row_name(row)
      for row in docker.list_managed_containers(
          image_filter, name_prefix, include_all=True
      )
  }
  if all_ or not names:
    targets = sorted(managed)
  else:
    targets = [resolve_name(name, name_prefix) for name in names]

  removed, skipped = [], []
  for name in targets:
    if name not in managed:
      skipped.append(f'{name} (not a managed container)')
      continue
    try:
      docker.docker_rm(name, force=force)
      removed.append(name)
    except docker.DockerCommandError as exc:
      skipped.append(f'{name} ({exc})')
  return removed, skipped


def restart_container(
    name: str,
    *,
    image: str | None = None,
    name_prefix: str = constants.DEFAULT_NAME_PREFIX,
    detach: bool = True,
) -> list[str]:
  """Recreates a container (``stop`` -> ``rm`` -> ``run``).

  The android_world server is PID 1 inside the container, so there is no child
  server process to restart in place; a full recreate is the only option, as in
  ``server/restart_aw.sh``. Returns the new ``docker run`` command executed.
  """
  info = get_container_info(name, name_prefix)
  port = info.host_port if info and info.host_port else _port_from_name(
      name, name_prefix
  )
  if port is None:
    raise ValueError(f'cannot determine host port for {name!r}')
  used_image = (info.image if info and info.image else None) or image or (
      constants.DEFAULT_IMAGE
  )
  cmd = docker.build_run_command(
      name=name, image=used_image, host_port=port, detach=detach
  )
  docker.docker_stop(name)
  docker.docker_rm(name)
  docker.run_command(cmd)
  return cmd


def check_prerequisites(
    image: str = constants.DEFAULT_IMAGE,
) -> list[constants.CheckResult]:
  """Runs read-only prerequisite checks and returns their results."""
  docker_path = shutil.which('docker')
  if not docker_path:
    return [
        constants.CheckResult(
            name='Docker installed', passed=False,
            message='docker not found on PATH', hint='Install Docker Engine',
        )
    ]

  version = docker.docker_version()
  results = [
      constants.CheckResult(
          name='Docker installed', passed=version.returncode == 0,
          message=(version.stdout or version.stderr).strip()
          or f'found at {docker_path}',
      )
  ]

  info = docker.docker_info()
  stderr = (info.stderr or '').lower()
  daemon_ok = info.returncode == 0
  results.append(
      constants.CheckResult(
          name='Docker daemon', passed=daemon_ok,
          message='docker info succeeded' if daemon_ok else 'docker info failed',
          hint='' if daemon_ok else 'sudo systemctl start docker',
      )
  )
  permission_denied = 'permission denied' in stderr
  results.append(
      constants.CheckResult(
          name='Docker permissions', passed=not permission_denied,
          message='permission denied' if permission_denied else 'no permission errors',
          hint=(
              'sudo usermod -aG docker $USER && newgrp docker'
              if permission_denied else ''
          ),
      )
  )

  kvm = '/dev/kvm'
  kvm_present = os.path.exists(kvm)
  kvm_rw = kvm_present and os.access(kvm, os.R_OK | os.W_OK)
  results.append(
      constants.CheckResult(
          name='KVM available', passed=kvm_present,
          message=(
              '/dev/kvm present (read/write)' if kvm_rw
              else '/dev/kvm present (check group permissions)' if kvm_present
              else '/dev/kvm not found'
          ),
          hint=(
              '' if kvm_rw else
              'sudo usermod -aG kvm $USER; containers run with --privileged'
          ),
      )
  )

  image_ok = docker.docker_image_inspect(image)
  results.append(
      constants.CheckResult(
          name=f'Image {image}', passed=image_ok,
          message='present locally' if image_ok else 'not found locally',
          hint='' if image_ok else f'docker build -t {image} .',
      )
  )
  return results
