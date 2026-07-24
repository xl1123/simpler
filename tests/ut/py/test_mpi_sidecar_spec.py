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


def _sidecar_spec(path: Path, *, rank: int = 1) -> RemoteWorkerSpec:
    return RemoteWorkerSpec(
        endpoint="127.0.0.1:19073",
        platform="a2a3sim",
        control_transport="mpi_sidecar",
        sidecar_endpoint=str(path),
        mpi_rank=rank,
    )


def test_remote_worker_spec_keeps_socket_defaults():
    spec = RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim")
    assert spec.control_transport == "socket"
    assert spec.sidecar_endpoint is None
    assert spec.mpi_rank is None


@pytest.mark.parametrize("control_transport", ["", "mpi", "tcp", "MPI_SIDECAR"])
def test_remote_worker_spec_rejects_unknown_control_transport(control_transport):
    with pytest.raises(ValueError, match="control_transport"):
        RemoteWorkerSpec(
            endpoint="127.0.0.1:19073", platform="a2a3sim", control_transport=control_transport
        )


@pytest.mark.parametrize(
    ("sidecar_endpoint", "mpi_rank", "match"),
    [
        (None, 1, "sidecar_endpoint"),
        ("relative.sock", 1, "absolute"),
        ("/tmp/sidecar.sock", None, "mpi_rank"),
        ("/tmp/sidecar.sock", -1, "mpi_rank"),
    ],
)
def test_mpi_sidecar_spec_requires_explicit_local_endpoint_and_rank(sidecar_endpoint, mpi_rank, match):
    with pytest.raises(ValueError, match=match):
        RemoteWorkerSpec(
            endpoint="127.0.0.1:19073",
            platform="a2a3sim",
            control_transport="mpi_sidecar",
            sidecar_endpoint=sidecar_endpoint,
            mpi_rank=mpi_rank,
        )


@pytest.mark.parametrize(
    ("sidecar_endpoint", "mpi_rank"),
    [("/tmp/sidecar.sock", None), (None, 1), ("/tmp/sidecar.sock", 1)],
)
def test_socket_spec_rejects_sidecar_only_fields(sidecar_endpoint, mpi_rank):
    with pytest.raises(ValueError, match="socket control transport"):
        RemoteWorkerSpec(
            endpoint="127.0.0.1:19073",
            platform="a2a3sim",
            sidecar_endpoint=sidecar_endpoint,
            mpi_rank=mpi_rank,
        )


def test_add_remote_worker_still_validates_remote_daemon_endpoint_for_sidecar(tmp_path):
    worker = Worker(level=4, num_sub_workers=0)
    try:
        with pytest.raises(ValueError, match="numeric IP"):
            worker.add_remote_worker(
                RemoteWorkerSpec(
                    endpoint="remote-host:19073",
                    platform="a2a3sim",
                    control_transport="mpi_sidecar",
                    sidecar_endpoint=str(tmp_path / "sidecar.sock"),
                    mpi_rank=1,
                )
            )
    finally:
        worker.close()


def test_open_remote_sidecar_session_forwards_manifest_and_rank(tmp_path):
    sidecar_path = tmp_path / "bootstrap.sock"
    command_path = tmp_path / "command.sock"
    health_path = tmp_path / "health.sock"
    received: dict[str, object] = {}
    ready = threading.Event()

    def serve():
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(sidecar_path))
        listener.listen(1)
        ready.set()
        conn, _ = listener.accept()
        with conn:
            size = struct.unpack("<I", _recv_exact(conn, 4))[0]
            request = json.loads(_recv_exact(conn, size).decode("utf-8"))
            received.update(request)
            reply = json.dumps(
                {
                    "ok": True,
                    "command_path": str(command_path),
                    "health_path": str(health_path),
                    "pid": 1234,
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
        session = worker._open_remote_sidecar_session(
            spec=_sidecar_spec(sidecar_path, rank=3),
            worker_id=7,
            session_id=11,
            deadline=time.monotonic() + 5.0,
        )
    finally:
        worker.close()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert received["version"] == 1
    assert received["op"] == "OPEN_SESSION"
    assert received["target_rank"] == 3
    assert received["daemon_host"] == "127.0.0.1"
    assert received["daemon_port"] == 19073
    manifest = received["manifest"]
    assert isinstance(manifest, dict)
    assert manifest["worker_id"] == 7
    assert manifest["session_id"] == 11
    assert manifest["session_timeout_s"] == 17.0
    assert 0 < manifest["startup_remaining_s"] <= 5.0
    assert session.command_path == str(command_path)
    assert session.health_path == str(health_path)
    assert session.pid == 1234


def test_activate_remote_sessions_selects_sidecar_attach(monkeypatch, tmp_path):
    clock = 1000.0
    monkeypatch.setattr(worker_mod.time, "monotonic", lambda: clock)
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=23.0)
    worker_id = worker.add_remote_worker(_sidecar_spec(tmp_path / "bootstrap.sock", rank=2))
    worker._worker = MagicMock()
    session = worker_mod._RemoteSidecarSession(
        worker_id=worker_id,
        session_id=9,
        command_path=str(tmp_path / "command.sock"),
        health_path=str(tmp_path / "health.sock"),
        pid=0,
    )
    open_sidecar = MagicMock(return_value=session)
    open_socket = MagicMock(side_effect=AssertionError("socket bootstrap selected for mpi_sidecar"))
    monkeypatch.setattr(worker, "_open_remote_sidecar_session", open_sidecar)
    monkeypatch.setattr(worker, "_open_remote_session", open_socket)

    try:
        worker._activate_remote_sessions(clock + 5.0)
        open_sidecar.assert_called_once()
        worker._worker.add_remote_l3_sidecar.assert_called_once_with(
            worker_id,
            9,
            "sim",
            str(tmp_path / "command.sock"),
            str(tmp_path / "health.sock"),
            5.0,
            23.0,
        )
        assert worker._worker.add_remote_l3_socket.call_count == 0
    finally:
        worker._remote_sessions.clear()
        worker._worker = None
        worker.close()


def test_expired_deadline_opens_no_sidecar_session(monkeypatch, tmp_path):
    worker = Worker(level=4, num_sub_workers=0)
    worker.add_remote_worker(_sidecar_spec(tmp_path / "bootstrap.sock"))
    worker._worker = MagicMock()
    opened = MagicMock()
    monkeypatch.setattr(worker, "_open_remote_sidecar_session", opened)
    try:
        with pytest.raises(RuntimeError, match="startup deadline exceeded"):
            worker._activate_remote_sessions(time.monotonic() - 1.0)
        opened.assert_not_called()
    finally:
        worker._worker = None
        worker.close()
