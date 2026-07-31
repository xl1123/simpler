# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import simpler.worker as worker_mod
from simpler.mpi_l3_gateway import Envelope, Lane, MessageType, MpiL3Gateway, decode_envelope
from simpler.mpi_l3_worker import _launcher_rank
from simpler.worker import Worker


class _RecordingTransport:
    def __init__(self) -> None:
        self.sent: list[tuple[int, bytes]] = []

    def send(self, rank: int, payload: bytes) -> None:
        self.sent.append((rank, payload))

    def receive(self) -> tuple[int, bytes] | None:
        return None


def _controller(tmp_path: Path) -> MpiL3Gateway:
    return MpiL3Gateway(
        rank=0,
        world_size=2,
        bootstrap_socket=str(tmp_path / "gateway.sock"),
        session_dir=str(tmp_path / "sessions"),
        inner_worker=object(),  # type: ignore[arg-type]
        rank_config={
            "platform": "a2a3sim",
            "runtime": "tensormap_and_ringbuffer",
            "device_ids": [0, 1],
        },
    )


def test_rank_zero_routes_local_worker_without_mpi_self_send(tmp_path, monkeypatch):
    controller = _controller(tmp_path)
    transport = _RecordingTransport()
    dispatched = []
    monkeypatch.setattr(controller, "_dispatch", dispatched.append)
    envelope = Envelope(MessageType.OPEN_SESSION, -1, 0, session_id=7)

    controller._route_outbound(transport, envelope)  # type: ignore[arg-type]

    assert transport.sent == []
    assert dispatched == [Envelope(MessageType.OPEN_SESSION, 0, 0, session_id=7)]


def test_rank_zero_sends_remote_worker_envelope_through_mpi(tmp_path):
    controller = _controller(tmp_path)
    transport = _RecordingTransport()
    envelope = Envelope(
        MessageType.FRAME_L4_TO_L3,
        -1,
        1,
        session_id=7,
        lane=Lane.COMMAND,
        sequence=9,
        payload=b"frame",
    )

    controller._route_outbound(transport, envelope)  # type: ignore[arg-type]

    assert len(transport.sent) == 1
    target, payload = transport.sent[0]
    assert target == 1
    assert decode_envelope(payload) == Envelope(
        MessageType.FRAME_L4_TO_L3,
        0,
        1,
        session_id=7,
        lane=Lane.COMMAND,
        sequence=9,
        payload=b"frame",
    )


def test_launcher_rank_reads_mpich_metadata(monkeypatch):
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    monkeypatch.setenv("PMI_RANK", "1")
    monkeypatch.setenv("PMI_SIZE", "2")
    assert _launcher_rank() == (1, 2)


def test_mpi_init_seam_runs_after_forks_and_before_scheduler(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    native = MagicMock()
    events = []
    native.init.side_effect = lambda: events.append("scheduler")
    worker._worker = native
    worker._startup_deadline = time.monotonic() + 5.0
    monkeypatch.setattr(worker_mod, "Orchestrator", lambda *_args: object())
    try:
        worker._start_hierarchical(_post_fork_pre_threads=lambda: events.append("mpi"))
        assert events == ["mpi", "scheduler"]
    finally:
        worker._worker = None
        worker.close()


def test_direct_launcher_derives_mpich_hostfile(tmp_path):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "mpi_l3_launch", root / "tools/mpi_l3/launch.py"
    )
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    hosts = [
        {"rank": 0, "mpi_host": "120.9.10.37"},
        {"rank": 1, "mpi_host": "120.9.10.35"},
    ]

    command = launcher._mpi_command(
        {"mpi": {"implementation": "mpich", "launcher": "/opt/mpich/bin/mpirun"}},
        hosts,
        tmp_path,
    )

    assert command == [
        "/opt/mpich/bin/mpirun",
        "-f",
        str(tmp_path / "mpi.hostfile"),
        "-ppn",
        "1",
        "-np",
        "2",
    ]
    assert (tmp_path / "mpi.hostfile").read_text(encoding="utf-8") == (
        "120.9.10.37\n120.9.10.35\n"
    )


def test_direct_launcher_compares_command_frame_identity(tmp_path, capsys):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "mpi_l3_launch_log", root / "tools/mpi_l3/launch.py"
    )
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    records = [
        {"event": "MPI_INIT_AFTER_L2_FORK"},
        {"event": "MPI_INIT_AFTER_L2_FORK"},
        {"event": "MPI_L3_WORLD_READY"},
        {"event": "MPI_L3_WORLD_READY"},
        {"event": "L4_MPI_L3_GATEWAY_READY"},
        {"event": "MPI_L3_SESSION_READY"},
        {"event": "MPI_L3_SESSION_READY"},
        {"event": "MPI_L3_SEND"},
        {"event": "MPI_L3_RECV"},
    ]
    task = {
        "lane": "COMMAND",
        "session_id": 7,
        "worker_id": 1,
        "frame_type": 2,
        "sequence": 3,
        "sha256": "a",
    }
    completion = {
        "lane": "COMMAND",
        "session_id": 7,
        "worker_id": 1,
        "frame_type": 5,
        "sequence": 3,
        "sha256": "b",
    }
    records.extend(
        [
            {"event": "L4_TO_MPI", **task},
            {"event": "MPI_TO_L3", **task},
            {"event": "L3_TO_MPI", **completion},
            {"event": "MPI_TO_L4", **completion},
        ]
    )
    log = tmp_path / "mpi-l3.log"
    log.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")

    launcher._verify_log(log)

    assert "command frame sequence/hash comparison PASS" in capsys.readouterr().out
