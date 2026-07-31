# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MPI-launched L3 rank with a rank-0 UDS gateway for a Simpler L4."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import signal
import sys
from typing import Any

from .mpi_l3_gateway import (
    MPI_L3_HEADER,
    MPI_L3_MAX_PAYLOAD,
    MpiL3Gateway,
    _log_event,
    stop_world,
)
from .worker import Worker

_ERROR_BYTES = 4096


def _launcher_rank() -> tuple[int, int]:
    for rank_name, size_name in (
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
        ("PMI_RANK", "PMI_SIZE"),
        ("PMIX_RANK", "PMIX_SIZE"),
    ):
        if rank_name in os.environ and size_name in os.environ:
            return int(os.environ[rank_name]), int(os.environ[size_name])
    raise RuntimeError("mpirun did not publish OMPI/PMI rank and world-size metadata")


class MpiTransport:
    """ctypes wrapper; every method must be called by the rank main thread."""

    def __init__(self, library: str) -> None:
        self._library_path = os.path.abspath(library)
        self._lib = ctypes.CDLL(self._library_path)
        u8_ptr = ctypes.POINTER(ctypes.c_uint8)
        char_ptr = ctypes.POINTER(ctypes.c_char)
        self._lib.simpler_mpi_l3_init.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            char_ptr,
            ctypes.c_uint64,
            char_ptr,
            ctypes.c_uint64,
        ]
        self._lib.simpler_mpi_l3_init.restype = ctypes.c_int
        self._lib.simpler_mpi_l3_send.argtypes = [
            ctypes.c_int,
            u8_ptr,
            ctypes.c_uint64,
            char_ptr,
            ctypes.c_uint64,
        ]
        self._lib.simpler_mpi_l3_send.restype = ctypes.c_int
        self._lib.simpler_mpi_l3_probe.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64),
            char_ptr,
            ctypes.c_uint64,
        ]
        self._lib.simpler_mpi_l3_probe.restype = ctypes.c_int
        self._lib.simpler_mpi_l3_recv.argtypes = [
            ctypes.c_int,
            u8_ptr,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            char_ptr,
            ctypes.c_uint64,
        ]
        self._lib.simpler_mpi_l3_recv.restype = ctypes.c_int
        for name in ("simpler_mpi_l3_barrier", "simpler_mpi_l3_finalize"):
            function = getattr(self._lib, name)
            function.argtypes = [char_ptr, ctypes.c_uint64]
            function.restype = ctypes.c_int
        self._lib.simpler_mpi_l3_abort.argtypes = [ctypes.c_int]
        self._lib.simpler_mpi_l3_abort.restype = None
        self.initialized = False

    @staticmethod
    def _error_buffer() -> ctypes.Array[ctypes.c_char]:
        return ctypes.create_string_buffer(_ERROR_BYTES)

    @staticmethod
    def _raise(operation: str, error: ctypes.Array[ctypes.c_char]) -> None:
        detail = error.value.decode("utf-8", errors="replace") or "unknown MPI error"
        raise RuntimeError(f"{operation}: {detail}")

    def initialize(self, *, rank: int, world_size: int, topology_id: str) -> str:
        error = self._error_buffer()
        processor = ctypes.create_string_buffer(256)
        result = self._lib.simpler_mpi_l3_init(
            rank,
            world_size,
            topology_id.encode("utf-8"),
            processor,
            len(processor),
            error,
            len(error),
        )
        if result != 0:
            # The C bridge may have initialized MPI before a topology check
            # failed. Mark it active so abort/finalize can tear down that world.
            self.initialized = True
            self._raise("MPI_Init_thread", error)
        self.initialized = True
        return processor.value.decode("utf-8", errors="replace")

    def send(self, target_rank: int, payload: bytes) -> None:
        data = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        error = self._error_buffer()
        if self._lib.simpler_mpi_l3_send(target_rank, data, len(payload), error, len(error)) != 0:
            self._raise("MPI_Isend", error)

    def receive(self) -> tuple[int, bytes] | None:
        source = ctypes.c_int()
        size = ctypes.c_uint64()
        error = self._error_buffer()
        available = self._lib.simpler_mpi_l3_probe(
            ctypes.byref(source), ctypes.byref(size), error, len(error)
        )
        if available < 0:
            self._raise("MPI_Iprobe", error)
        if available == 0:
            return None
        maximum = MPI_L3_HEADER.size + MPI_L3_MAX_PAYLOAD
        if size.value <= 0 or size.value > maximum:
            raise RuntimeError(f"MPI envelope size {size.value} exceeds protocol limit {maximum}")
        data = (ctypes.c_uint8 * size.value)()
        actual = ctypes.c_uint64()
        if (
            self._lib.simpler_mpi_l3_recv(
                source.value, data, size.value, ctypes.byref(actual), error, len(error)
            )
            != 0
        ):
            self._raise("MPI_Recv", error)
        if actual.value != size.value:
            raise RuntimeError("MPI probe/receive envelope length changed")
        return source.value, bytes(data)

    def barrier(self) -> None:
        error = self._error_buffer()
        if self._lib.simpler_mpi_l3_barrier(error, len(error)) != 0:
            self._raise("MPI_Barrier", error)

    def finalize(self) -> None:
        if not self.initialized:
            return
        error = self._error_buffer()
        if self._lib.simpler_mpi_l3_finalize(error, len(error)) != 0:
            self._raise("MPI_Finalize", error)
        self.initialized = False

    def abort(self, error_code: int = 1) -> None:
        if self.initialized:
            self._lib.simpler_mpi_l3_abort(error_code)



def _load_config(path: str, rank: int, world_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    hosts = config.get("hosts")
    if not isinstance(hosts, list) or len(hosts) != world_size:
        raise ValueError("topology hosts must contain one entry per MPI rank")
    by_rank = {int(item["rank"]): item for item in hosts}
    if set(by_rank) != set(range(world_size)):
        raise ValueError("topology hosts must contain every MPI rank exactly once")
    rank_config = dict(by_rank[rank])
    rank_config["platform"] = str(config.get("platform", "a2a3"))
    rank_config["runtime"] = str(config.get("runtime", "tensormap_and_ringbuffer"))
    if not isinstance(rank_config.get("device_ids"), list) or not rank_config["device_ids"]:
        raise ValueError(f"MPI rank {rank} requires non-empty device_ids")
    return config, rank_config


def run_rank(config_path: str, library: str, bootstrap_socket: str, session_dir: str) -> int:
    rank, world_size = _launcher_rank()
    config, rank_config = _load_config(config_path, rank, world_size)
    if world_size != 2:
        raise ValueError("direct MPI L3 phase requires exactly two ranks")
    transport = MpiTransport(library)
    worker = Worker(
        level=3,
        platform=rank_config["platform"],
        runtime=rank_config["runtime"],
        device_ids=[int(item) for item in rank_config["device_ids"]],
        num_sub_workers=int(rank_config.get("num_sub_workers", 0)),
        startup_timeout_s=float(config.get("timeout_s", 300.0)),
    )
    processor_name = ""

    def initialize_mpi() -> None:
        nonlocal processor_name
        _log_event("MPI_INIT_AFTER_L2_FORK", rank=rank, l3_pid=os.getpid())
        processor_name = transport.initialize(
            rank=rank,
            world_size=world_size,
            topology_id=str(config.get("topology_id", "two-host-direct-l3")),
        )

    try:
        worker.init(_post_fork_pre_threads=initialize_mpi)
        transport.barrier()
        _log_event(
            "MPI_L3_WORLD_READY",
            rank=rank,
            world_size=world_size,
            host=processor_name,
            l3_pid=os.getpid(),
            device_ids=rank_config["device_ids"],
        )
        controller = MpiL3Gateway(
            rank=rank,
            world_size=world_size,
            bootstrap_socket=bootstrap_socket if rank == 0 else None,
            session_dir=session_dir,
            inner_worker=worker,
            rank_config=rank_config,
        )
        return controller.serve(
            transport,
            {
                "event": "MPI L3 world READY",
                "rank": rank,
                "world_size": world_size,
                "host": processor_name,
                "l3_pid": os.getpid(),
            },
        )
    except BaseException:
        transport.abort(1)
        raise
    finally:
        with contextlib.suppress(BaseException):
            worker.close()
        if transport.initialized:
            transport.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--mpi-library")
    parser.add_argument("--bootstrap-socket")
    parser.add_argument("--session-dir")
    parser.add_argument("--stop-world")
    ns = parser.parse_args(argv)
    if ns.stop_world:
        return stop_world(ns.stop_world)
    if not all((ns.config, ns.mpi_library, ns.bootstrap_socket, ns.session_dir)):
        parser.error("--config, --mpi-library, --bootstrap-socket and --session-dir are required")

    def terminate(_signum, _frame):
        raise SystemExit(128 + signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate)
    return run_rank(ns.config, ns.mpi_library, ns.bootstrap_socket, ns.session_dir)


if __name__ == "__main__":
    sys.exit(main())
