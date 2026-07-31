#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the mpirun L4->two L3->four L2 compute+communication smoke on 35/37."""

from __future__ import annotations

import argparse
import struct
import sys

from compute_then_tload_smoke import (
    COUNT,
    FLOAT_BYTES,
    MAX_RANKS,
    WINDOW_SIZE,
    _digest_scalars,
    _expected_communication,
    _expected_compute,
    _lhs_values,
    _max_diff,
    _rhs_values,
    _unpack_floats,
    build_communication_callable,
    build_compute_callable,
)
from simpler.task_interface import CallConfig, CommBufferSpec, TaskArgs
from simpler.worker import MpiL3GroupSpec, RemoteCallable, Worker


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one value")
    return values


def _remote_args(domain_id: int, local_worker_id: int, chip_digest: bytes) -> TaskArgs:
    args = TaskArgs()
    args.add_scalar(domain_id)
    args.add_scalar(local_worker_id)
    for value in _digest_scalars(chip_digest):
        args.add_scalar(value)
    return args


def run(args: argparse.Namespace) -> int:
    devices37 = _parse_csv_ints(args.devices_37)
    devices35 = _parse_csv_ints(args.devices_35)
    roce37 = _parse_csv_strings(args.roce_37)
    roce35 = _parse_csv_strings(args.roce_35)
    if len(devices37) != 2 or len(devices35) != 2:
        raise ValueError("this smoke is intentionally fixed to 2+2 devices")
    if len(roce37) != len(devices37) or len(roce35) != len(devices35):
        raise ValueError("RoCE IP count must match device count on each server")

    rank_count = len(devices37) + len(devices35)
    if not 2 <= rank_count <= MAX_RANKS:
        raise ValueError(f"the smoke requires between 2 and {MAX_RANKS} global ranks")

    mpirun_args = tuple(args.mpirun_arg) if args.mpirun_arg else ("--map-by", "ppr:1:node")
    print("[mpirun-2x2] management hosts:")
    print(f"  rank0/L3 on 37: {args.host_37}, devices={devices37}, roce={roce37}")
    print(f"  rank1/L3 on 35: {args.host_35}, devices={devices35}, roce={roce35}")
    print("[mpirun-2x2] RoCE IPs are logged here for operator verification; Fabric descriptors are exported by L2.")

    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=args.remote_session_timeout_s)
    node37, node35 = worker.add_mpirun_worker_group(
        MpiL3GroupSpec(
            hosts=(f"{args.host_37}:1", f"{args.host_35}:1"),
            command_port_base=args.command_port_base,
            health_port_base=args.health_port_base,
            device_ids_by_rank=(devices37, devices35),
            platform=args.platform,
            runtime=args.runtime,
            comm_profile="a3-fabric-v1",
            global_device_ranks_by_rank=((0, 1), (2, 3)),
            session_listen_hosts=(args.host_37, args.host_35),
            connect_hosts=(args.host_37, args.host_35),
            ready_host=args.ready_host or args.host_37,
            ready_port=args.ready_port,
            mpirun_path=args.mpirun_path,
            mpirun_args=mpirun_args,
            python_executable=args.python_executable,
        )
    )
    node_layout = ((node37, devices37), (node35, devices35))
    targets = tuple(
        (node_id, local_worker_id) for node_id, device_ids in node_layout for local_worker_id in range(len(device_ids))
    )

    print(f"[mpirun-2x2] compiling for {args.platform}/{args.runtime}")
    compute_handle = worker.register(build_compute_callable(args.platform, args.runtime))
    communication_handle = worker.register(build_communication_callable(args.platform, args.runtime))
    remote_compute_handle = worker.register(
        RemoteCallable("simpler.global_comm_smoke:remote_compute_orch"),
        workers=[node37, node35],
    )
    remote_communication_handle = worker.register(
        RemoteCallable("simpler.global_comm_smoke:remote_rank_orch"),
        workers=[node37, node35],
    )

    captured: dict[str, object] = {}
    observed_compute: list[tuple[float, ...]] = []
    observed_communication: list[tuple[float, ...]] = []
    try:
        worker.init()

        def compute_phase(orch, _args, cfg):
            domain = orch.allocate_global_domain(
                name="a3-l4-mpirun-2x2-compute-then-tload",
                members=targets,
                window_size=WINDOW_SIZE,
                buffers=(
                    CommBufferSpec("lhs", "float32", COUNT, COUNT * FLOAT_BYTES),
                    CommBufferSpec("rhs", "float32", COUNT, COUNT * FLOAT_BYTES),
                    CommBufferSpec("input", "float32", COUNT, COUNT * FLOAT_BYTES),
                    CommBufferSpec("result", "float32", COUNT, COUNT * FLOAT_BYTES),
                ),
                retain_after_run=True,
            )
            for domain_rank, (node_id, local_worker_id) in enumerate(targets):
                orch.copy_to_global_domain(
                    domain,
                    domain_rank,
                    struct.pack(f"<{COUNT}f", *_lhs_values(domain_rank)),
                    buffer="lhs",
                )
                orch.copy_to_global_domain(
                    domain,
                    domain_rank,
                    struct.pack(f"<{COUNT}f", *_rhs_values(domain_rank)),
                    buffer="rhs",
                )
                orch.submit_next_level(
                    remote_compute_handle,
                    _remote_args(domain.domain_id, local_worker_id, compute_handle.digest),
                    cfg,
                    worker=node_id,
                )
            captured["domain"] = domain

        worker.run(compute_phase, args=None, config=CallConfig())
        domain = captured["domain"]

        def communication_phase(orch, _args, cfg):
            for domain_rank, (node_id, local_worker_id) in enumerate(targets):
                raw = orch.copy_from_global_domain(
                    domain,
                    domain_rank,
                    COUNT * FLOAT_BYTES,
                    buffer="input",
                )
                observed_compute.append(_unpack_floats(raw))
                orch.submit_next_level(
                    remote_communication_handle,
                    _remote_args(domain.domain_id, local_worker_id, communication_handle.digest),
                    cfg,
                    worker=node_id,
                )

        worker.run(communication_phase, args=None, config=CallConfig())

        def verify_phase(orch, _args, _cfg):
            try:
                for domain_rank in range(len(targets)):
                    raw = orch.copy_from_global_domain(
                        domain,
                        domain_rank,
                        COUNT * FLOAT_BYTES,
                        buffer="result",
                    )
                    observed_communication.append(_unpack_floats(raw))
            finally:
                domain.release()

        worker.run(verify_phase, args=None, config=CallConfig())

        for domain_rank, result in enumerate(observed_compute):
            max_diff = _max_diff(result, _expected_compute(domain_rank))
            print(f"[mpirun-2x2] compute domain_rank={domain_rank} max_diff={max_diff:.3e}")
            if max_diff > 1e-5:
                raise AssertionError(f"domain_rank {domain_rank} compute golden mismatch: max_diff={max_diff}")

        expected_communication = _expected_communication(len(targets))
        for domain_rank, result in enumerate(observed_communication):
            max_diff = _max_diff(result, expected_communication)
            print(f"[mpirun-2x2] communication domain_rank={domain_rank} max_diff={max_diff:.3e}")
            if max_diff > 1e-3:
                raise AssertionError(f"domain_rank {domain_rank} communication golden mismatch: max_diff={max_diff}")

        print("[mpirun-2x2] PASS: two L3 ranks exchanged descriptors by MPI and four L2 ranks completed TLOAD")
        return 0
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-37", default="120.9.10.37")
    parser.add_argument("--host-35", default="120.9.10.35")
    parser.add_argument("--roce-37", default="10.30.2.1,10.30.2.2")
    parser.add_argument("--roce-35", default="10.30.0.1,10.30.0.2")
    parser.add_argument("--devices-37", default="0,1")
    parser.add_argument("--devices-35", default="0,1")
    parser.add_argument("--command-port-base", type=int, default=26000)
    parser.add_argument("--health-port-base", type=int, default=26100)
    parser.add_argument("--ready-host", default="")
    parser.add_argument("--ready-port", type=int, default=0)
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--remote-session-timeout-s", type=float, default=180.0)
    parser.add_argument("--mpirun-path", default="mpirun")
    parser.add_argument("--mpirun-arg", action="append", default=[])
    parser.add_argument("--python-executable", default=sys.executable)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
