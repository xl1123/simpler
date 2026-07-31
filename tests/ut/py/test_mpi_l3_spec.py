# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import simpler.worker as worker_mod
from simpler.worker import RemoteWorkerSpec, Worker


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("test peer closed")
        data.extend(chunk)
    return bytes(data)


def _mpi_l3_spec(path: Path, *, rank: int = 1) -> RemoteWorkerSpec:
    return RemoteWorkerSpec(
        endpoint=f"mpi://rank/{rank}",
        platform="a2a3sim",
        control_transport="mpi_l3",
        gateway_endpoint=str(path),
        mpi_rank=rank,
    )


def test_remote_worker_spec_keeps_socket_defaults():
    spec = RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim")
    assert spec.control_transport == "socket"
    assert spec.gateway_endpoint is None
    assert spec.mpi_rank is None


@pytest.mark.parametrize("control_transport", ["", "mpi", "tcp", "MPI_L3"])
def test_remote_worker_spec_rejects_unknown_control_transport(control_transport):
    with pytest.raises(ValueError, match="control_transport"):
        RemoteWorkerSpec(
            endpoint="127.0.0.1:19073", platform="a2a3sim", control_transport=control_transport
        )


@pytest.mark.parametrize(
    ("endpoint", "gateway", "rank", "match"),
    [
        ("mpi://rank/1", None, 1, "gateway_endpoint"),
        ("mpi://rank/1", "relative.sock", 1, "absolute"),
        ("mpi://rank/1", "/tmp/gateway.sock", None, "mpi_rank"),
        ("mpi://rank/1", "/tmp/gateway.sock", -1, "mpi_rank"),
        ("127.0.0.1:19073", "/tmp/gateway.sock", 1, "mpi://rank/1"),
    ],
)
def test_mpi_l3_spec_rejects_ambiguous_route(endpoint, gateway, rank, match):
    with pytest.raises(ValueError, match=match):
        RemoteWorkerSpec(
            endpoint=endpoint,
            platform="a2a3sim",
            control_transport="mpi_l3",
            gateway_endpoint=gateway,
            mpi_rank=rank,
        )


@pytest.mark.parametrize(
    ("gateway", "rank"),
    [("/tmp/gateway.sock", None), (None, 1), ("/tmp/gateway.sock", 1)],
)
def test_socket_spec_rejects_mpi_fields(gateway, rank):
    with pytest.raises(ValueError, match="socket control transport"):
        RemoteWorkerSpec(
            endpoint="127.0.0.1:19073",
            platform="a2a3sim",
            gateway_endpoint=gateway,
            mpi_rank=rank,
        )


def test_mpi_l3_spec_uses_logical_rank_and_local_gateway(tmp_path):
    spec = _mpi_l3_spec(tmp_path / "gateway.sock", rank=1)
    assert spec.endpoint == "mpi://rank/1"
    assert spec.gateway_endpoint == str(tmp_path / "gateway.sock")

    worker = Worker(level=4, num_sub_workers=0)
    try:
        assert worker.add_remote_worker(spec) == 0
    finally:
        worker.close()


def test_mpi_l3_spec_rejects_tcp_session_listener_fields(tmp_path):
    with pytest.raises(ValueError, match="TCP session listener"):
        RemoteWorkerSpec(
            endpoint="mpi://rank/1",
            platform="a2a3sim",
            control_transport="mpi_l3",
            gateway_endpoint=str(tmp_path / "gateway.sock"),
            mpi_rank=1,
            session_listen_host="0.0.0.0",
            allow_wildcard_session_bind=True,
        )


def test_open_mpi_l3_session_has_no_tcp_daemon_fields(tmp_path):
    gateway_path = tmp_path / "gateway.sock"
    command_path = tmp_path / "command.sock"
    health_path = tmp_path / "health.sock"
    received: dict[str, object] = {}
    ready = threading.Event()

    def serve():
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(gateway_path))
        listener.listen(1)
        ready.set()
        conn, _ = listener.accept()
        with conn:
            size = struct.unpack("<I", _recv_exact(conn, 4))[0]
            received.update(json.loads(_recv_exact(conn, size).decode("utf-8")))
            reply = json.dumps(
                {
                    "ok": True,
                    "command_path": str(command_path),
                    "health_path": str(health_path),
                    "pid": 4321,
                },
                sort_keys=True,
            ).encode("utf-8")
            conn.sendall(struct.pack("<I", len(reply)) + reply)
        listener.close()

    thread = threading.Thread(target=serve)
    thread.start()
    assert ready.wait(timeout=2.0)
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=17.0)
    try:
        session = worker._open_remote_mpi_l3_session(
            spec=_mpi_l3_spec(gateway_path, rank=1),
            worker_id=1,
            session_id=11,
            deadline=time.monotonic() + 5.0,
        )
    finally:
        worker.close()
        thread.join(timeout=2.0)

    assert received["target_rank"] == 1
    assert "daemon_host" not in received
    assert "daemon_port" not in received
    manifest = received["manifest"]
    assert isinstance(manifest, dict)
    assert "listen_host" not in manifest
    assert "connect_host" not in manifest
    assert session.command_path == str(command_path)
    assert session.health_path == str(health_path)


def test_activate_remote_sessions_selects_mpi_l3_without_tcp(monkeypatch, tmp_path):
    clock = 1000.0
    monkeypatch.setattr(worker_mod.time, "monotonic", lambda: clock)
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=23.0)
    worker_id = worker.add_remote_worker(_mpi_l3_spec(tmp_path / "gateway.sock", rank=0))
    worker._worker = MagicMock()
    session = worker_mod._RemoteUnixSession(
        worker_id=worker_id,
        session_id=9,
        command_path=str(tmp_path / "command.sock"),
        health_path=str(tmp_path / "health.sock"),
        pid=0,
    )
    open_mpi = MagicMock(return_value=session)
    open_tcp = MagicMock(side_effect=AssertionError("TCP bootstrap selected for mpi_l3"))
    monkeypatch.setattr(worker, "_open_remote_mpi_l3_session", open_mpi)
    monkeypatch.setattr(worker, "_open_remote_session", open_tcp)

    try:
        worker._activate_remote_sessions(clock + 5.0)
        open_mpi.assert_called_once()
        worker._worker.add_remote_l3_unix.assert_called_once_with(
            worker_id,
            9,
            "sim",
            str(tmp_path / "command.sock"),
            str(tmp_path / "health.sock"),
            5.0,
            23.0,
        )
        open_tcp.assert_not_called()
        worker._worker.add_remote_l3_socket.assert_not_called()
    finally:
        worker._remote_sessions.clear()
        worker._worker = None
        worker.close()


def test_expired_deadline_opens_no_mpi_l3_session(monkeypatch, tmp_path):
    worker = Worker(level=4, num_sub_workers=0)
    worker.add_remote_worker(_mpi_l3_spec(tmp_path / "gateway.sock"))
    worker._worker = MagicMock()
    opened = MagicMock()
    monkeypatch.setattr(worker, "_open_remote_mpi_l3_session", opened)
    try:
        with pytest.raises(RuntimeError, match="startup deadline exceeded"):
            worker._activate_remote_sessions(time.monotonic() - 1.0)
        opened.assert_not_called()
    finally:
        worker._worker = None
        worker.close()
