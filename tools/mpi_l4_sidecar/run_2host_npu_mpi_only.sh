#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
config=${1:?usage: run_2host_npu_mpi_only.sh topology.2host-npu.json}
exec "${PYTHON:-python3}" "${repo_root}/tools/mpi_l4_sidecar/launch.py" \
  --mode npu --mpi-only --config "${config}"
