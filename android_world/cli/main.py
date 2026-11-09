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

"""Entry point for the ``aw`` console script."""

import argparse
import logging
import sys

from android_world.cli import env


def create_parser() -> argparse.ArgumentParser:
  """Builds the top-level parser."""
  parser = argparse.ArgumentParser(
      prog='aw',
      description='android_world command line interface',
  )
  subparsers = parser.add_subparsers(dest='command', help='Available commands')
  env.configure_parser(subparsers)
  return parser


def _set_log_level(args: argparse.Namespace) -> int:
  """Maps ``-q`` / ``-v`` onto the stdlib logging level.

  Unlike the reference ``mw env`` implementation, these flags are actually
  consumed: ``-v`` surfaces the docker commands executed at DEBUG level.
  """
  level = logging.WARNING if getattr(args, 'quiet', False) else logging.INFO
  level -= 10 * getattr(args, 'verbose', 0)
  level = max(logging.DEBUG, min(logging.CRITICAL, level))
  logging.basicConfig(level=level, format='%(levelname)s %(message)s')
  return level


def main() -> None:
  """Parses arguments and dispatches to the ``env`` command group."""
  parser = create_parser()
  args = parser.parse_args()
  _set_log_level(args)
  if args.command == 'env':
    env.execute(args)
  else:
    parser.print_help()
    sys.exit(1)


if __name__ == '__main__':
  main()
