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

"""The ``aw env`` subcommand group: argparse registration and rich output.

This is the presentation layer. All environment logic lives in
:mod:`android_world.cli.api`; all docker access lives in
:mod:`android_world.cli.docker`.
"""

import argparse
import json
import shlex
import sys
from dataclasses import asdict

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from android_world.cli import api
from android_world.cli import constants
from android_world.cli import docker

console = Console()


def _esc(value: object) -> str:
  """Escapes rich markup in dynamic values coming from docker."""
  return escape(str(value))


def _write_json(payload: object) -> None:
  """Writes machine-readable JSON to stdout without rich decoration."""
  sys.stdout.write(json.dumps(payload, indent=2) + '\n')


def _fmt_health(value: bool | None) -> str:
  if value is None:
    return '[dim]-[/dim]'
  return '[green]healthy[/green]' if value else '[yellow]booting / unreachable[/yellow]'


def _add_common_options(parser: argparse.ArgumentParser, *, image=False,
                        prefix=False) -> None:
  """Adds the shared identification flags with hyphen/underscore aliases."""
  if prefix:
    parser.add_argument(
        '--name-prefix', '--name_prefix', '--prefix', dest='name_prefix',
        default=constants.DEFAULT_NAME_PREFIX,
        help='Container name prefix (default: %(default)s)',
    )
  if image:
    parser.add_argument(
        '--image', default=constants.DEFAULT_IMAGE_FILTER,
        help='Image substring used to identify managed containers '
             '(default: %(default)s)',
    )


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
  """Registers the ``env`` command group and all of its subcommands."""
  env_parser = subparsers.add_parser(
      'env', help='Manage Docker environments for android_world',
  )
  env_parser.add_argument(
      '-q', '--quiet', action='store_true',
      help='Reduce output (only warnings and errors)',
  )
  env_parser.add_argument(
      '-v', '--verbose', action='count', default=0,
      help='Increase verbosity (-v for DEBUG, -vv for TRACE)',
  )
  env_subparsers = env_parser.add_subparsers(
      dest='env_action', help='Environment management actions', required=True,
  )

  # run
  run_parser = env_subparsers.add_parser(
      'run', help='Launch Docker container(s)',
  )
  run_parser.add_argument(
      '--port', type=int, default=constants.DEFAULT_HOST_PORT,
      help='Base host port (default: %(default)s)',
  )
  run_parser.add_argument(
      '--count', type=int, default=1,
      help='Number of containers to launch (default: %(default)s)',
  )
  _add_common_options(run_parser, prefix=True)
  run_parser.add_argument(
      '--image', default=constants.DEFAULT_IMAGE,
      help='Image to launch (default: %(default)s)',
  )
  run_parser.add_argument(
      '--dry-run', '--dry_run', dest='dry_run', action='store_true',
      help='Print the exact docker run command without executing it',
  )
  run_parser.add_argument(
      '--wait', action=argparse.BooleanOptionalAction, default=False,
      help='Wait for /health after launching (default: --no-wait)',
  )
  run_parser.add_argument(
      '--detach', action=argparse.BooleanOptionalAction, default=True,
      help='Run detached (default: --detach)',
  )

  # rm
  rm_parser = env_subparsers.add_parser(
      'rm', help='Destroy Docker container(s)',
  )
  rm_parser.add_argument(
      'container_names', nargs='*',
      help='Container names or numeric host ports',
  )
  rm_parser.add_argument(
      '-a', '--all', dest='all_', action='store_true',
      help='Remove all managed containers',
  )
  rm_parser.add_argument(
      '-f', '--force', action='store_true',
      help='Force removal of running containers (docker rm -f)',
  )
  _add_common_options(rm_parser, image=True, prefix=True)

  # list / ls
  list_parser = env_subparsers.add_parser(
      'list', aliases=['ls'],
      help='List running android_world containers',
  )
  list_parser.add_argument(
      '-a', '--all', dest='all_', action='store_true',
      help='Include stopped containers',
  )
  list_parser.add_argument(
      '--json', dest='as_json', action='store_true',
      help='Emit JSON instead of a table',
  )
  list_parser.add_argument(
      '--health', action='store_true',
      help='Probe GET /health for each container',
  )
  _add_common_options(list_parser, image=True, prefix=True)

  # info
  info_parser = env_subparsers.add_parser(
      'info', help='Get detailed info about a container',
  )
  info_parser.add_argument(
      'container_name', help='Container name or numeric host port',
  )
  info_parser.add_argument(
      '--json', dest='as_json', action='store_true',
      help='Emit JSON instead of a table',
  )
  info_parser.add_argument(
      '--health', action='store_true', help='Probe GET /health',
  )
  _add_common_options(info_parser, prefix=True)

  # restart
  restart_parser = env_subparsers.add_parser(
      'restart', help='Restart the android_world server in a container',
  )
  restart_parser.add_argument(
      'container_name', help='Container name or numeric host port',
  )
  restart_parser.add_argument(
      '-i', '--interactive', action='store_true',
      help='Show the docker commands used for the recreate',
  )
  _add_common_options(restart_parser, prefix=True)

  # exec
  exec_parser = env_subparsers.add_parser(
      'exec', help='Open a bash shell in a container',
  )
  exec_parser.add_argument(
      'container_name', help='Container name or numeric host port',
  )
  exec_parser.add_argument(
      '-c', '--command', dest='command', default='/bin/bash',
      help='Command to run inside the container (default: %(default)s)',
  )
  _add_common_options(exec_parser, prefix=True)

  # check
  check_parser = env_subparsers.add_parser(
      'check',
      help='Check prerequisites for running android_world (Docker, KVM)',
  )
  check_parser.add_argument(
      '--health', action='store_true',
      help='Also probe /health on running managed containers',
  )
  check_parser.add_argument(
      '--json', dest='as_json', action='store_true',
      help='Emit JSON instead of a table',
  )
  _add_common_options(check_parser, image=True, prefix=True)


def _run(args: argparse.Namespace) -> None:
  ports = api.find_available_ports(args.count, args.port)
  if len(ports) < args.count:
    console.print(
        f'[yellow]requested {args.count} port(s), only {len(ports)} '
        'free port(s) available[/yellow]'
    )
  if not ports:
    console.print('[red]no free host ports available[/red]')
    return

  if args.dry_run:
    results = api.launch_containers(
        ports, image=args.image, name_prefix=args.name_prefix,
        detach=args.detach, dry_run=True,
    )
    table = Table(
        title='Dry run — planned containers', box=box.ROUNDED,
        header_style='bold magenta',
    )
    table.add_column('Name')
    table.add_column('Port')
    table.add_column('Command')
    for result in results:
      table.add_row(
          _esc(result.name), str(result.port),
          _esc(shlex.join(result.command or [])),
      )
    console.print(table)
    console.print(Panel(
        '[dim]Dry run only — no container was created.[/dim]', title='Note',
    ))
    return

  with Progress(
      SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
      TimeElapsedColumn(), transient=True,
  ) as progress:
    progress.add_task(f'launching {len(ports)} container(s)…', total=None)
    results = api.launch_containers(
        ports, image=args.image, name_prefix=args.name_prefix,
        detach=args.detach, wait=args.wait,
    )

  if getattr(args, 'as_json', False):
    _write_json([asdict(result) for result in results])
    return

  table = Table(
      title='Launch results', box=box.ROUNDED, header_style='bold magenta',
  )
  table.add_column('Name')
  table.add_column('Port')
  table.add_column('Status')
  table.add_column('Detail')
  for result in results:
    status = result.status
    if status == 'launched':
      status = '[green]launched[/green]'
    elif status in ('already_exists', 'image_missing', 'failed'):
      status = f'[yellow]{status}[/yellow]'
    table.add_row(
        _esc(result.name), str(result.port), status, _esc(result.message),
    )
  console.print(table)


def _rm(args: argparse.Namespace) -> None:
  removed, skipped = api.remove_containers(
      args.container_names, all_=args.all_, force=args.force,
      image_filter=args.image, name_prefix=args.name_prefix,
  )
  if getattr(args, 'as_json', False):
    _write_json({'removed': removed, 'skipped': skipped})
    return
  console.print(Panel(
      f'[green]removed {len(removed)}[/green] · '
      f'[yellow]skipped {len(skipped)}[/yellow]',
      title='Remove results',
  ))
  if removed:
    console.print('removed: ' + ', '.join(_esc(n) for n in removed))
  if skipped:
    console.print('[dim]skipped: ' + '; '.join(_esc(s) for s in skipped) + '[/dim]')


def _collect_health(infos: list[constants.ContainerInfo]) -> dict[str, bool]:
  """Probes /health for each container, showing a rich progress spinner."""
  health: dict[str, bool] = {}
  with Progress(
      SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
      TimeElapsedColumn(), transient=True,
  ) as progress:
    task = progress.add_task('probing /health…', total=len(infos))
    for info in infos:
      health[info.name] = (
          api.probe_health(info.host_port) if info.host_port else False
      )
      progress.advance(task)
  return health


def _list(args: argparse.Namespace) -> None:
  infos = api.list_containers(
      image_filter=args.image, name_prefix=args.name_prefix,
      include_all=args.all_,
  )
  health = _collect_health(infos) if args.health else {}

  if args.as_json:
    payload = []
    for info in infos:
      entry = asdict(info)
      entry['health'] = health.get(info.name)
      payload.append(entry)
    _write_json(payload)
    return

  table = Table(
      title='android_world containers', box=box.ROUNDED,
      header_style='bold magenta',
  )
  table.add_column('Name')
  table.add_column('Image')
  table.add_column('Status')
  table.add_column('Host Port')
  table.add_column('Created')
  if args.health:
    table.add_column('Health')
  for info in infos:
    row = [
        _esc(info.name), _esc(info.image), _esc(info.status),
        f'http://localhost:{info.host_port}' if info.host_port else 'N/A',
        _esc(info.created_at),
    ]
    if args.health:
      row.append(_fmt_health(health.get(info.name)))
    table.add_row(*row)
  console.print(table)
  if not infos:
    console.print('[dim]no managed containers found[/dim]')


def _info(args: argparse.Namespace) -> None:
  name = api.resolve_name(args.container_name, args.name_prefix)
  info = api.get_container_info(name)
  if info is None:
    console.print(f'[red]container {_esc(name)} not found[/red]')
    sys.exit(1)
  health = (
      api.probe_health(info.host_port)
      if args.health and info.host_port else None
  )

  if args.as_json:
    payload = asdict(info)
    payload['health'] = health
    _write_json(payload)
    return

  props = Table(
      title=f'Container: {info.name}', box=box.ROUNDED,
      header_style='bold magenta', show_header=False,
  )
  props.add_column('Property')
  props.add_column('Value')
  props.add_row('Name', _esc(info.name))
  props.add_row('Image', _esc(info.image))
  props.add_row('Status', _esc(info.status))
  props.add_row('Running', '[green]yes[/green]' if info.running else '[red]no[/red]')
  props.add_row('State', _esc(info.state))
  props.add_row('Started At', _esc(info.started_at))
  props.add_row('Created At', _esc(info.created_at))
  props.add_row('Restart Policy', _esc(info.restart_policy))
  props.add_row('Memory', _esc(info.memory))
  console.print(props)

  ports = Table(
      title='Port mappings', box=box.ROUNDED, header_style='bold magenta',
  )
  ports.add_column('Container')
  ports.add_column('Host')
  ports.add_row(
      f'{constants.CONTAINER_PORT}/tcp',
      f'http://localhost:{info.host_port}' if info.host_port else 'N/A',
  )
  if args.health:
    ports.add_row('Health', _fmt_health(health))
  console.print(ports)


def _restart(args: argparse.Namespace) -> None:
  name = api.resolve_name(args.container_name, args.name_prefix)
  console.print(Panel(
      f'[bold yellow]This STOPS and REMOVES {_esc(name)} and recreates it. '
      'The Android emulator reboots (≈5–10 minutes).[/bold yellow]',
      title='Restart warning',
  ))
  if args.interactive:
    console.print(
        '[dim]docker stop → docker rm → docker run (--init --privileged)  '
        '[/dim]'
    )
  cmd = api.restart_container(name)
  console.print(f'[green]recreated {_esc(name)}[/green]')
  console.print(f'[dim]{_esc(shlex.join(cmd))}[/dim]')


def _exec(args: argparse.Namespace) -> None:
  name = api.resolve_name(args.container_name, args.name_prefix)
  info = api.get_container_info(name)
  if info is None or not info.running:
    console.print(f'[red]container {_esc(name)} is not running[/red]')
    sys.exit(1)
  console.print(
      f'[dim]entering {_esc(name)} — the server is PID 1; do not stop it or '
      'call /close.[/dim]'
  )
  docker.docker_exec_replace(name, args.command)


def _check(args: argparse.Namespace) -> None:
  results = api.check_prerequisites()
  if args.health:
    infos = api.list_containers(
        image_filter=args.image, name_prefix=args.name_prefix,
    )
    for info in infos:
      if not info.running:
        continue
      ok = api.probe_health(info.host_port) if info.host_port else False
      results.append(constants.CheckResult(
          name=f'Health {info.name}', passed=ok,
          message='healthy' if ok else 'booting / unreachable',
          hint='' if ok else f'docker logs {info.name}',
      ))

  if args.as_json:
    _write_json([asdict(result) for result in results])
    return

  table = Table(
      title='Prerequisites', box=box.ROUNDED, header_style='bold magenta',
  )
  table.add_column('Check')
  table.add_column('Status')
  table.add_column('Message')
  for result in results:
    status = '[green]✓ PASS[/green]' if result.passed else '[red]✗ FAIL[/red]'
    table.add_row(_esc(result.name), status, _esc(result.message))
  console.print(table)
  for result in results:
    if not result.passed and result.hint:
      console.print(f'  [yellow]hint: {_esc(result.hint)}[/yellow]')

  failed = sum(not result.passed for result in results)
  if failed:
    console.print(Panel(f'[red]{failed} check(s) failed[/red]', title='Check'))
    sys.exit(1)
  console.print(Panel('[green]all checks passed[/green]', title='Check'))


def execute(args: argparse.Namespace) -> None:
  """Dispatches an ``env`` subcommand to its handler (synchronous)."""
  action_map = {
      'run': _run,
      'rm': _rm,
      'list': _list,
      'ls': _list,
      'info': _info,
      'restart': _restart,
      'exec': _exec,
      'check': _check,
  }
  action = action_map.get(args.env_action)
  if action is None:
    raise ValueError(f'unknown env action: {args.env_action!r}')
  action(args)
