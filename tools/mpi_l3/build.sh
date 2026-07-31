#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir=${1:-"${repo_root}/build/mpi_l3_transport"}

cmake -S "${repo_root}/src/common/hierarchical/mpi_l3_transport" -B "${build_dir}"
cmake --build "${build_dir}" -j2
echo "${build_dir}/libsimpler_mpi_l3_transport.so"
