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

"""Setup file for AndroidWorld."""

import os

import pkg_resources
import setuptools
from setuptools.command import build_py

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PROTOS = (
    'android_world/task_evals/information_retrieval/proto/state.proto',
    'android_world/task_evals/information_retrieval/proto/task.proto',
)


class _GenerateProtoFiles(setuptools.Command):
  """Command to generate protobuf bindings for AndroidEnv protos."""

  descriptions = 'Generates Python protobuf bindings for AndroidEnv protos.'
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    # Import grpc_tools here, after setuptools has installed setup_requires
    # dependencies.
    from grpc_tools import protoc  # pylint: disable=g-import-not-at-top

    grpc_protos_include = pkg_resources.resource_filename(
        'grpc_tools', '_proto'
    )

    for proto_path in _PACKAGE_PROTOS:
      proto_args = [
          'grpc_tools.protoc',
          '--proto_path={}'.format(grpc_protos_include),
          '--proto_path={}'.format(_ROOT_DIR),
          '--python_out={}'.format(_ROOT_DIR),
          '--grpc_python_out={}'.format(_ROOT_DIR),
          os.path.join(_ROOT_DIR, proto_path),
      ]
      if protoc.main(proto_args) != 0:
        raise RuntimeError('ERROR: {}'.format(proto_args))


class _BuildPy(build_py.build_py):
  """Generate protobuf bindings during the build_py stage."""

  def run(self):
    self.run_command('generate_protos')
    super().run()

_GRPCIO_TOOLS_VERSION = '1.71.0'
_PROTOBUF_VERSION = '5.29.5'

# NOTE: Do NOT call `setuptools.find_packages()` without a `where` argument.
# Inside the Docker image the project is copied to `/` (`WORKDIR /` +
# `COPY . /`), so the implicit `where="."` resolves to `/` and setuptools
# os.walk()s the ENTIRE container filesystem with followlinks=True, recursing
# infinitely into `/proc/self/task/*/cwd` (a procfs symlink pointing back to
# `/`). Scope the search to the package directory only.
# `find_packages()` returns names relative to `where`, so re-prefix the root
# package name to obtain the full dotted paths.
_PACKAGE_ROOT = os.path.join(_ROOT_DIR, 'android_world')
_PACKAGES = ['android_world'] + [
    'android_world.' + package
    for package in setuptools.find_packages(where=_PACKAGE_ROOT)
]

setuptools.setup(
    name='android_world',
    package_data={
        '': [
            '*.json',
            '*.proto',
            '*.textproto',
            '*.xml',
            'res/xml/*.xml',
        ]
    },
    packages=_PACKAGES,
    setup_requires=[f'grpcio-tools=={_GRPCIO_TOOLS_VERSION}'],
    install_requires=[
        f'protobuf=={_PROTOBUF_VERSION}',
    ],
    cmdclass={
        'build_py': _BuildPy,
        'generate_protos': _GenerateProtoFiles,
    },
)
