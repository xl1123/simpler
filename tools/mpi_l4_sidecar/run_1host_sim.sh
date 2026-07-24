#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
config=${1:-"${repo_root}/tools/mpi_l4_sidecar/topology.local.json"}
exec "${PYTHON:-python3}" "${repo_root}/tools/mpi_l4_sidecar/launch.py" --mode one-host --config "${config}"
