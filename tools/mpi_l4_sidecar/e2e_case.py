# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared legacy-socket/MPI-sidecar L4 -> L3 -> L2 sim validation case."""

from __future__ import annotations

import argparse
import json
import sys

from simpler.callable_identity import build_python_import_descriptor, compute_callable_hashid, hashid_to_digest
from simpler.remote_l3_session import get_inner_handle
from simpler.task_interface import TaskArgs
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

INNER_TARGET = "tools.mpi_l4_sidecar.e2e_case:inner_l2_check"
INNER_HASHID = compute_callable_hashid(build_python_import_descriptor(*INNER_TARGET.split(":")))


def inner_l2_check(args):
    if args.scalar_count() != 1 or args.scalar(0) != 17:
        raise RuntimeError("MPI sidecar L2 scalar mismatch")


def remote_l3_submit_l2(orch, _args, _cfg):
    args = TaskArgs()
    args.add_scalar(17)
    orch.submit_sub(get_inner_handle(INNER_HASHID), args)


def run_case(ns: argparse.Namespace) -> dict[str, object]:
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=ns.timeout)
    worker_id = -1
    inner_committed = False
    try:
        kwargs: dict[str, object] = {}
        if ns.control_transport == "mpi_sidecar":
            kwargs.update(
                control_transport="mpi_sidecar",
                sidecar_endpoint=ns.sidecar_endpoint,
                mpi_rank=ns.mpi_rank,
            )
        worker_id = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=ns.daemon_endpoint,
                platform=ns.platform,
                transport="sim",
                num_sub_workers=1,
                **kwargs,
            )
        )
        handle = worker.register(
            RemoteCallable("tools.mpi_l4_sidecar.e2e_case:remote_l3_submit_l2"), workers=[worker_id]
        )
        worker.init()
        print(
            json.dumps(
                {
                    "event": "L4 HELLO READY",
                    "control_transport": ns.control_transport,
                    "worker_id": worker_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        assert worker._worker is not None
        inner_digest = hashid_to_digest(INNER_HASHID)
        result = worker._worker.remote_prepare_register(
            worker_id,
            "INNER_L3_WORKER",
            "PYTHON_IMPORT",
            INNER_TARGET.encode("utf-8"),
            inner_digest,
        )
        if not result.ok:
            raise RuntimeError(result.error_message)
        result = worker._worker.remote_commit_register(
            worker_id,
            "INNER_L3_WORKER",
            "PYTHON_IMPORT",
            inner_digest,
        )
        if not result.ok:
            raise RuntimeError(result.error_message)
        inner_committed = True

        def parent_orch(orch, _args, cfg):
            orch.submit_next_level(handle, TaskArgs(), cfg, worker=worker_id)

        worker.run(parent_orch)
        result = worker._worker.remote_unregister(
            worker_id,
            "INNER_L3_WORKER",
            "PYTHON_IMPORT",
            inner_digest,
        )
        if not result.ok:
            raise RuntimeError(result.error_message)
        inner_committed = False
        result_record = {
            "status": "PASS",
            "control_transport": ns.control_transport,
            "worker_id": worker_id,
            "remote_l3": True,
            "inner_l2": True,
            "golden": 17,
        }
        print(
            json.dumps(
                {
                    "event": "L4 -> remote L3 -> L2 TASK/COMPLETION PASS",
                    **result_record,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return result_record
    finally:
        if inner_committed and worker._worker is not None:
            try:
                worker._worker.remote_unregister(
                    worker_id,
                    "INNER_L3_WORKER",
                    "PYTHON_IMPORT",
                    hashid_to_digest(INNER_HASHID),
                )
            except Exception:  # noqa: BLE001
                pass
        worker.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-transport", choices=("socket", "mpi_sidecar"), required=True)
    parser.add_argument("--daemon-endpoint", required=True)
    parser.add_argument("--sidecar-endpoint")
    parser.add_argument("--mpi-rank", type=int, default=1)
    parser.add_argument("--platform", default="a2a3sim")
    parser.add_argument("--timeout", type=float, default=30.0)
    ns = parser.parse_args(argv)
    if ns.control_transport == "mpi_sidecar" and not ns.sidecar_endpoint:
        parser.error("--sidecar-endpoint is required for mpi_sidecar")
    print(json.dumps(run_case(ns), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
