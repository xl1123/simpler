# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import socket
import struct
import threading
import time

import pytest

from simpler.remote_l3_sidecar_proxy import (
    Envelope,
    Lane,
    MessageType,
    SidecarProxy,
    _slr3_identity,
    _worker_ids_for_rank,
    encode_envelope,
    read_envelope,
    stop_world,
)


def _frame(*, session_id: int = 9, worker_id: int = 3, sequence: int = 7, payload: bytes = b"abc") -> bytes:
    return b"SLR3" + struct.pack("<IIQiQII", 1, 2, session_id, worker_id, sequence, len(payload), 0) + payload


def test_sidecar_envelope_roundtrip_over_stream_socket():
    left, right = socket.socketpair()
    try:
        expected = Envelope(MessageType.FRAME, 0, 1, 9, Lane.COMMAND, 7, _frame())
        left.sendall(encode_envelope(expected))
        assert read_envelope(right) == expected
    finally:
        left.close()
        right.close()


def test_sidecar_envelope_rejects_oversize_before_send():
    envelope = Envelope(MessageType.FRAME, 0, 1, payload=b"x" * (16 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="exceeds maximum"):
        encode_envelope(envelope)


def test_sidecar_envelope_reports_truncated_stream():
    left, right = socket.socketpair()
    left.sendall(encode_envelope(Envelope(MessageType.SHUTDOWN, 0, 0))[:10])
    left.close()
    try:
        with pytest.raises(EOFError, match="socket closed"):
            read_envelope(right)
    finally:
        right.close()


def test_slr3_identity_is_preserved_without_reencoding():
    frame = _frame(session_id=17, worker_id=4, sequence=99)
    assert _slr3_identity(frame) == (17, 4, 99)


def test_slr3_identity_rejects_payload_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        _slr3_identity(_frame()[:-1])


def test_worker_map_requires_rank_and_exact_worker_set():
    assert _worker_ids_for_rank("0:;1:0,2;2:1", 0) == set()
    assert _worker_ids_for_rank("0:;1:0,2;2:1", 1) == {0, 2}
    with pytest.raises(ValueError, match="no entry"):
        _worker_ids_for_rank("0:;1:0", 2)
    with pytest.raises(ValueError, match="duplicate rank"):
        _worker_ids_for_rank("0:;0:1", 0)
    with pytest.raises(ValueError, match="duplicate worker"):
        _worker_ids_for_rank("0:1,1", 0)


def test_multiple_envelopes_keep_stream_boundaries():
    left, right = socket.socketpair()
    envelopes = [
        Envelope(MessageType.OPEN_SESSION, 0, 1, 1, payload=b"one"),
        Envelope(MessageType.FRAME, 1, 0, 1, Lane.HEALTH, 2, _frame(session_id=1, sequence=2)),
    ]

    def writer():
        left.sendall(b"".join(encode_envelope(item) for item in envelopes))
        left.close()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert [read_envelope(right), read_envelope(right)] == envelopes
    finally:
        right.close()
        thread.join(timeout=2)


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("condition was not met")


def test_proxy_publishes_l4_bootstrap_only_after_mpi_world_ready(tmp_path):
    sidecar_path = tmp_path / "proxy.0.sock"
    bootstrap_path = tmp_path / "bootstrap.sock"
    proxy = SidecarProxy(
        rank=0,
        sidecar_socket=str(sidecar_path),
        session_dir=str(tmp_path / "sessions"),
        worker_ids=set(),
        bootstrap_socket=str(bootstrap_path),
    )
    result: list[int] = []
    thread = threading.Thread(target=lambda: result.append(proxy.serve()))
    thread.start()
    _wait_until(sidecar_path.exists)
    assert not bootstrap_path.exists()

    sidecar = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sidecar.connect(str(sidecar_path))
    world = b'{"event":"MPI world READY","rank":0,"world_size":2,"worker_map":"0:;1:0","hosts":["a","b"]}'
    sidecar.sendall(encode_envelope(Envelope(MessageType.WORLD_READY, 0, 0, payload=world)))
    _wait_until(bootstrap_path.exists)

    stop_result: list[int] = []
    stop_thread = threading.Thread(target=lambda: stop_result.append(stop_world(str(bootstrap_path))))
    stop_thread.start()
    shutdown = read_envelope(sidecar)
    assert shutdown.message_type == MessageType.SHUTDOWN
    assert shutdown.source_rank == -1
    assert shutdown.target_rank == 0
    sidecar.sendall(encode_envelope(Envelope(MessageType.SHUTDOWN, 0, 0)))

    stop_thread.join(timeout=2)
    thread.join(timeout=2)
    sidecar.close()
    assert not stop_thread.is_alive()
    assert not thread.is_alive()
    assert stop_result == [0]
    assert result == [0]
    assert not sidecar_path.exists()
    assert not bootstrap_path.exists()
