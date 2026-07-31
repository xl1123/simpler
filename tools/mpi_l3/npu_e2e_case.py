# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the real L4 -> MPI L3 -> two-L2-NPU validation task."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import sys
from pathlib import Path

from simpler.callable_identity import build_chip_callable_descriptor, compute_callable_hashid, hashid_to_digest
from simpler.remote_l3_protocol import (
    CallableKind,
    ChipCallableBlobLocation,
    RemoteChipCallablePayload,
    encode_remote_chip_callable_payload,
)
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, DataType, RemoteTensorRef, TaskArgs
from simpler.task_interface import TensorArgType
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

REMOTE_ORCH_TARGET = "tools.mpi_l3.npu_e2e_case:remote_l3_group_orch"
ELEMENTS = 128 * 128
FLOAT_NBYTES = ctypes.sizeof(ctypes.c_float)
TENSOR_COUNT = 6
FloatArray = ctypes.c_float * ELEMENTS
_REMOTE_GROUP_KEEPALIVE: list[TaskArgs] = []


def remote_l3_group_orch(orch, args, cfg):
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    digest = b"".join(int(args.scalar(i)).to_bytes(8, "little", signed=False) for i in range(4))
    chip_handle = get_inner_handle(digest.hex())
    chip_args = []
    for offset in (0, 3):
        task_args = TaskArgs()
        task_args.add_tensor(args.tensor(offset), TensorArgType.INPUT)
        task_args.add_tensor(args.tensor(offset + 1), TensorArgType.INPUT)
        task_args.add_tensor(args.tensor(offset + 2), TensorArgType.OUTPUT_EXISTING)
        chip_args.append(task_args)
    _REMOTE_GROUP_KEEPALIVE[:] = chip_args
    orch.submit_next_level_group(chip_handle, chip_args, cfg, workers=[0, 1])


def _build_vector_chip_callable(platform: str, runtime: str) -> ChipCallable:
    root = Path(__file__).resolve().parents[2]
    kernels = root / "examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = compiler.get_orchestration_include_dirs(runtime)
    orchestration_binary = compiler.compile_orchestration(
        runtime,
        str(kernels / "orchestration/example_orchestration.cpp"),
    )

    def compile_aiv(name: str) -> bytes:
        raw = compiler.compile_incore(
            str(kernels / f"aiv/{name}"),
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=include_dirs,
        )
        return raw if platform.endswith("sim") else extract_text_section(raw)

    children = [
        (0, CoreCallable.build(signature=[D.IN, D.IN, D.OUT], binary=compile_aiv("kernel_add.cpp"))),
        (1, CoreCallable.build(signature=[D.IN, D.OUT], binary=compile_aiv("kernel_add_scalar.cpp"))),
        (2, CoreCallable.build(signature=[D.IN, D.IN, D.OUT], binary=compile_aiv("kernel_mul.cpp"))),
    ]
    return ChipCallable.build(
        signature=[D.IN, D.IN, D.OUT],
        func_name="aicpu_orchestration_entry",
        binary=orchestration_binary,
        children=children,
        config_name="aicpu_orchestration_config",
    )


def _remote_chip_register_payload(chip: ChipCallable, *, platform: str, runtime: str) -> tuple[bytes, bytes]:
    blob = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    descriptor = build_chip_callable_descriptor(target=chip, platform=platform, runtime=runtime)
    digest = hashid_to_digest(compute_callable_hashid(descriptor))
    payload = encode_remote_chip_callable_payload(
        RemoteChipCallablePayload(
            descriptor_bytes=descriptor,
            blob_location=ChipCallableBlobLocation.INLINE_BLOB,
            blob_size=len(blob),
            blob_sha256=hashlib.sha256(blob).digest(),
            inline_blob=blob,
            staged_blob_token=b"",
        )
    )
    return digest, payload


def _parse_device_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(ids) != 2:
        raise ValueError("each remote L3 requires exactly two device IDs for the group case")
    return ids


def _make_array(value: float) -> FloatArray:
    array = FloatArray()
    for index in range(ELEMENTS):
        array[index] = value
    return array


def _expected(lhs: float, rhs: float) -> float:
    summed = lhs + rhs
    return (summed + 1.0) * (summed + 2.0) + summed


def _max_diff(array: FloatArray, expected: float) -> float:
    result = 0.0
    for value in array:
        actual = float(value)
        if not math.isfinite(actual):
            return math.inf
        result = max(result, abs(actual - expected))
    return result


def _add_digest_scalars(task_args: TaskArgs, digest: bytes) -> None:
    if len(digest) != 32:
        raise ValueError("inner chip callable digest must be 32 bytes")
    for offset in range(0, 32, 8):
        task_args.add_scalar(int.from_bytes(digest[offset : offset + 8], "little", signed=False))


def _make_remote_group_args(handles: list, digest: bytes) -> TaskArgs:
    args = TaskArgs()
    for index, handle in enumerate(handles):
        tag = TensorArgType.OUTPUT_EXISTING if index in (2, 5) else TensorArgType.INPUT
        args.add_tensor(RemoteTensorRef(handle, shape=(ELEMENTS,), dtype=DataType.FLOAT32), tag)
    _add_digest_scalars(args, digest)
    return args


def _install_inner_chip_callable(worker: Worker, worker_id: int, digest: bytes, payload: bytes) -> None:
    assert worker._worker is not None  # noqa: SLF001
    result = worker._worker.remote_prepare_register(  # noqa: SLF001
        worker_id,
        "INNER_L3_WORKER",
        CallableKind.CHIP_CALLABLE.name,
        payload,
        digest,
    )
    if not result.ok:
        raise RuntimeError(f"remote prepare inner CHIP_CALLABLE failed on worker {worker_id}: {result.error_message}")
    result = worker._worker.remote_commit_register(  # noqa: SLF001
        worker_id,
        "INNER_L3_WORKER",
        CallableKind.CHIP_CALLABLE.name,
        digest,
    )
    if not result.ok:
        raise RuntimeError(f"remote commit inner CHIP_CALLABLE failed on worker {worker_id}: {result.error_message}")


def _remote_spec(
    ns: argparse.Namespace, endpoint: str, devices: tuple[int, ...], mpi_rank: int
) -> RemoteWorkerSpec:
    transport_args: dict[str, object] = {}
    if ns.control_transport == "mpi_l3":
        transport_args = {
            "control_transport": "mpi_l3",
            "gateway_endpoint": ns.gateway_endpoint,
            "mpi_rank": mpi_rank,
        }
    return RemoteWorkerSpec(
        endpoint=endpoint,
        platform=ns.platform,
        runtime=ns.runtime,
        device_ids=devices,
        transport="sim",
        session_listen_host=ns.session_listen_host if ns.control_transport != "mpi_l3" else None,
        allow_wildcard_session_bind=ns.control_transport != "mpi_l3",
        **transport_args,
    )


def run_case(ns: argparse.Namespace) -> dict[str, object]:
    machine_a_devices = _parse_device_ids(ns.machine_a_devices)
    machine_b_devices = _parse_device_ids(ns.machine_b_devices)
    print(f"[npu-case:{ns.control_transport}] compiling vector kernels", flush=True)
    chip = _build_vector_chip_callable(ns.platform, ns.runtime)
    digest, payload = _remote_chip_register_payload(chip, platform=ns.platform, runtime=ns.runtime)
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=ns.timeout)
    remote_buffers = []
    parent_keepalive: list[TaskArgs] = []
    try:
        worker_a = worker.add_remote_worker(
            _remote_spec(ns, ns.machine_a, machine_a_devices, ns.machine_a_mpi_rank)
        )
        worker_b = worker.add_remote_worker(
            _remote_spec(ns, ns.machine_b, machine_b_devices, ns.machine_b_mpi_rank)
        )
        remote_handle = worker.register(RemoteCallable(REMOTE_ORCH_TARGET), workers=[worker_a, worker_b])
        print(f"[npu-case:{ns.control_transport}] initializing L4 and two remote L3 workers", flush=True)
        worker.init()
        print(f"[npu-case:{ns.control_transport}] installing L2 chip callable", flush=True)
        _install_inner_chip_callable(worker, worker_a, digest, payload)
        _install_inner_chip_callable(worker, worker_b, digest, payload)

        tensor_nbytes = ELEMENTS * FLOAT_NBYTES
        group_values = {
            worker_a: (2.0, 3.0, 4.0, 5.0),
            worker_b: (6.0, 7.0, 8.0, 9.0),
        }
        group_handles = {}
        output_arrays = {}
        for worker_id, values in group_values.items():
            handles = [worker.remote_malloc(worker=worker_id, nbytes=tensor_nbytes) for _ in range(TENSOR_COUNT)]
            remote_buffers.extend(handles)
            group_handles[worker_id] = handles
            input_arrays = [
                _make_array(values[0]),
                _make_array(values[1]),
                _make_array(0.0),
                _make_array(values[2]),
                _make_array(values[3]),
                _make_array(0.0),
            ]
            for handle, array in zip(handles, input_arrays):
                worker.remote_copy_to(handle, array, tensor_nbytes)
            output_arrays[worker_id] = {
                "group0": (_make_array(0.0), _expected(values[0], values[1])),
                "group1": (_make_array(0.0), _expected(values[2], values[3])),
            }

        def parent_orch(orch, _args, cfg):
            args_a = _make_remote_group_args(group_handles[worker_a], digest)
            args_b = _make_remote_group_args(group_handles[worker_b], digest)
            parent_keepalive[:] = [args_a, args_b]
            orch.submit_next_level(remote_handle, args_a, cfg, worker=worker_a)
            orch.submit_next_level(remote_handle, args_b, cfg, worker=worker_b)

        config = CallConfig()
        config.block_dim = ns.block_dim
        config.aicpu_thread_num = 4
        print(
            f"[npu-case:{ns.control_transport}] submitting cross-machine NPU task "
            f"with block_dim={ns.block_dim}",
            flush=True,
        )
        worker.run(parent_orch, config=config)

        print(f"[npu-case:{ns.control_transport}] reading and checking NPU outputs", flush=True)
        for worker_id, handles in group_handles.items():
            worker.remote_copy_from(handles[2], output_arrays[worker_id]["group0"][0], tensor_nbytes)
            worker.remote_copy_from(handles[5], output_arrays[worker_id]["group1"][0], tensor_nbytes)

        output_records = []
        max_diff = 0.0
        for worker_id, outputs in output_arrays.items():
            for group, (array, expected) in outputs.items():
                current_diff = _max_diff(array, expected)
                if current_diff > 1e-4:
                    raise AssertionError(f"worker {worker_id} {group} golden mismatch: max_diff={current_diff}")
                max_diff = max(max_diff, current_diff)
                output_records.append(
                    {
                        "worker_id": worker_id,
                        "group": group,
                        "expected": expected,
                        "sha256": hashlib.sha256(bytes(array)).hexdigest(),
                    }
                )
        return {
            "status": "PASS",
            "control_transport": ns.control_transport,
            "worker_ids": [worker_a, worker_b],
            "remote_l3": True,
            "inner_l2_npu": True,
            "elements": ELEMENTS,
            "block_dim": ns.block_dim,
            "max_diff": max_diff,
            "outputs": output_records,
        }
    finally:
        parent_keepalive.clear()
        for handle in reversed(remote_buffers):
            try:
                worker.remote_free(handle)
            except Exception:  # noqa: BLE001
                pass
        worker.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-transport", choices=("socket", "mpi_l3"), required=True)
    parser.add_argument("--machine-a", required=True)
    parser.add_argument("--machine-b", required=True)
    parser.add_argument("--machine-a-devices", default="0,1")
    parser.add_argument("--machine-b-devices", default="0,1")
    parser.add_argument("--machine-a-mpi-rank", type=int, default=0)
    parser.add_argument("--machine-b-mpi-rank", type=int, default=1)
    parser.add_argument("--gateway-endpoint")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--session-listen-host", default="0.0.0.0")
    parser.add_argument("--timeout", type=float, default=300.0)
    ns = parser.parse_args(argv)
    if ns.block_dim <= 0:
        parser.error("--block-dim must be positive")
    if ns.control_transport == "mpi_l3" and not ns.gateway_endpoint:
        parser.error("--gateway-endpoint is required for mpi_l3")
    print(json.dumps(run_case(ns), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
