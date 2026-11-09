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

"""Constants and data models shared by the ``aw env`` subcommands.

The values here mirror the conventions hardcoded in
``scripts/run_container.sh`` and ``server/restart_aw.sh`` so that the CLI and
the shell scripts launch identical containers.
"""

from dataclasses import dataclass

# Image used when launching a container. It is built locally, not pulled.
DEFAULT_IMAGE = 'android_world:latest'
# Tag-less substring used when matching containers. Keeping it tag-less means a
# container started from ``android_world:v2`` is still recognised.
DEFAULT_IMAGE_FILTER = 'android_world'
# All managed containers are named ``aw_<host_port>``.
DEFAULT_NAME_PREFIX = 'aw_'
# Default host port for ``aw env run``. The server inside the container always
# listens on CONTAINER_PORT.
DEFAULT_HOST_PORT = 3800
CONTAINER_PORT = 5000
MEMORY = '6g'
MEMORY_SWAP = '6g'
RESTART_POLICY = 'unless-stopped'
HEALTH_PATH = '/health'
HEALTH_TIMEOUT_SEC = 900
HEALTH_POLL_SEC = 5.0
HEALTH_PROBE_TIMEOUT_SEC = 2


@dataclass
class ContainerInfo:
  """Information about a managed container, used by ``list`` and ``info``."""

  name: str = ''
  image: str = ''
  status: str = ''
  running: bool = False
  host_port: int | None = None
  created_at: str = ''
  state: str = ''
  started_at: str = ''
  restart_policy: str = ''
  memory: str = ''


@dataclass
class LaunchResult:
  """Outcome of a single ``run`` slot.

  Attributes:
    name: Container name that was considered.
    port: Host port that was considered.
    status: One of ``launched``, ``already_exists``, ``image_missing``,
      ``dry_run``, ``failed``.
    command: The exact ``docker run`` argument list, when applicable.
    message: Human readable detail.
  """

  name: str
  port: int
  status: str
  command: list[str] | None = None
  message: str = ''


@dataclass
class CheckResult:
  """Outcome of a single ``check`` prerequisite."""

  name: str
  passed: bool
  message: str
  hint: str = ''
