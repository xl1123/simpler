#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.

set -euo pipefail

job_dir=${1:?usage: verify_no_residue.sh /tmp/simpler-mpi-l4-JOB}
if find "${job_dir}" -type s -print -quit | grep -q .; then
    echo "Unix socket residue remains under ${job_dir}" >&2
    find "${job_dir}" -type s -print >&2
    exit 1
fi
echo "no Unix socket residue under ${job_dir}"
