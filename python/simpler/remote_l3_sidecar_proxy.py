# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local proxy between a Simpler L4/L3 process and the MPI sidecar.

The proxy owns JSON bootstrap and socket lifecycle. The C++ sidecar only moves
opaque envelopes with MPI, so no MPI library is loaded into a Worker process.
"""

from __future__ import annotations

import argparse
import contextlib
import enum
import hashlib
import json
import os
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SIDECAR_MAGIC = b"SLM1"
SIDECAR_VERSION = 1
SIDECAR_HEADER = struct.Struct("<4sIIiiQIIQ")
SIDECAR_MAX_PAYLOAD = 16 * 1024 * 1024
SLR3_HEADER_BYTES = 40
SLR3_MAX_PAYLOAD = 16 * 1024 * 1024


class MessageType(enum.IntEnum):
    WORLD_READY = 1
    OPEN_SESSION = 2
    OPEN_SESSION_REPLY = 3
    FRAME = 4
    CLOSE_SESSION = 5
    ERROR = 6
    SHUTDOWN = 7


class Lane(enum.IntEnum):
    BOOTSTRAP = 0
    COMMAND = 1
    HEALTH = 2


@dataclass(frozen=True)
class Envelope:
    message_type: MessageType
    source_rank: int
    target_rank: int
    session_id: int = 0
    lane: Lane = Lane.BOOTSTRAP
    sequence: int = 0
    payload: bytes = b""


def encode_envelope(envelope: Envelope) -> bytes:
    payload = bytes(envelope.payload)
    if len(payload) > SIDECAR_MAX_PAYLOAD:
        raise ValueError("sidecar payload exceeds maximum")
    return SIDECAR_HEADER.pack(
        SIDECAR_MAGIC,
        SIDECAR_VERSION,
        int(envelope.message_type),
        int(envelope.source_rank),
        int(envelope.target_rank),
        int(envelope.session_id),
        int(envelope.lane),
        len(payload),
        int(envelope.sequence),
    ) + payload


def _read_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("socket closed")
        data.extend(chunk)
    return bytes(data)


def read_envelope(sock: socket.socket) -> Envelope:
    header = _read_exact(sock, SIDECAR_HEADER.size)
    magic, version, raw_type, source, target, session_id, raw_lane, payload_size, sequence = (
        SIDECAR_HEADER.unpack(header)
    )
    if magic != SIDECAR_MAGIC or version != SIDECAR_VERSION:
        raise ValueError("sidecar envelope magic or version mismatch")
    if payload_size > SIDECAR_MAX_PAYLOAD:
        raise ValueError("sidecar payload exceeds maximum")
    try:
        message_type = MessageType(raw_type)
        lane = Lane(raw_lane)
    except ValueError as exc:
        raise ValueError("sidecar envelope type or lane is unknown") from exc
    return Envelope(message_type, source, target, session_id, lane, sequence, _read_exact(sock, payload_size))


def _send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    if len(data) > SIDECAR_MAX_PAYLOAD:
        raise ValueError("JSON payload exceeds maximum")
    sock.sendall(struct.pack("<I", len(data)) + data)


def _read_json(sock: socket.socket) -> dict[str, Any]:
    size = struct.unpack("<I", _read_exact(sock, 4))[0]
    if size > SIDECAR_MAX_PAYLOAD:
        raise ValueError("JSON payload exceeds maximum")
    value = json.loads(_read_exact(sock, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON request must be an object")
    return value


def _slr3_identity(frame: bytes) -> tuple[int, int, int]:
    if len(frame) < SLR3_HEADER_BYTES or frame[:4] != b"SLR3":
        raise ValueError("invalid SLR3 frame header")
    version, _frame_type, session_id, worker_id, sequence, payload_size, flags = struct.unpack_from(
        "<IIQiQII", frame, 4
    )
    if version != 1 or flags != 0 or payload_size > SLR3_MAX_PAYLOAD:
        raise ValueError("invalid SLR3 frame metadata")
    if len(frame) != SLR3_HEADER_BYTES + payload_size:
        raise ValueError("SLR3 frame payload length mismatch")
    return int(session_id), int(worker_id), int(sequence)


def _log_frame(event: str, envelope: Envelope) -> None:
    _session_id, worker_id, sequence = _slr3_identity(envelope.payload)
    frame_type = struct.unpack_from("<I", envelope.payload, 8)[0]
    print(
        json.dumps(
            {
                "event": event,
                "rank": envelope.source_rank,
                "target_rank": envelope.target_rank,
                "session_id": envelope.session_id,
                "worker_id": worker_id,
                "lane": envelope.lane.name,
                "frame_type": frame_type,
                "sequence": sequence,
                "sha256": hashlib.sha256(envelope.payload).hexdigest(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _read_slr3(sock: socket.socket) -> bytes:
    header = _read_exact(sock, SLR3_HEADER_BYTES)
    payload_size = struct.unpack_from("<I", header, 32)[0]
    if payload_size > SLR3_MAX_PAYLOAD:
        raise ValueError("SLR3 frame payload exceeds maximum")
    frame = header + _read_exact(sock, payload_size)
    _slr3_identity(frame)
    return frame


def _bind_unix(path: str) -> socket.socket:
    if not os.path.isabs(path) or len(os.fsencode(path)) >= 104:
        raise ValueError("Unix socket path must be absolute and shorter than 104 encoded bytes")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    os.chmod(path, 0o600)
    listener.listen()
    return listener


def _worker_ids_for_rank(worker_map: str, rank: int) -> set[int]:
    result: dict[int, set[int]] = {}
    for entry in worker_map.split(";"):
        rank_text, separator, workers_text = entry.partition(":")
        if not separator or not rank_text:
            raise ValueError("worker map must use rank:worker,worker entries")
        parsed_rank = int(rank_text)
        if parsed_rank < 0 or parsed_rank in result:
            raise ValueError("worker map contains a duplicate rank")
        workers = [int(item) for item in workers_text.split(",") if item]
        if any(worker < 0 for worker in workers) or len(workers) != len(set(workers)):
            raise ValueError("worker map contains an invalid or duplicate worker id")
        result[parsed_rank] = set(workers)
    if rank not in result:
        raise ValueError(f"worker map has no entry for rank {rank}")
    return result[rank]


@dataclass
class _PendingOpen:
    event: threading.Event = field(default_factory=threading.Event)
    reply: dict[str, Any] | None = None


@dataclass
class _SourceSession:
    session_id: int
    worker_id: int
    target_rank: int
    command_path: str
    health_path: str
    command_listener: socket.socket
    health_listener: socket.socket
    sockets: dict[Lane, socket.socket] = field(default_factory=dict)
    queued: dict[Lane, list[bytes]] = field(default_factory=lambda: {Lane.COMMAND: [], Lane.HEALTH: []})
    lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        with self.lock:
            sockets = [self.command_listener, self.health_listener, *self.sockets.values()]
            self.sockets.clear()
        for sock in sockets:
            with contextlib.suppress(OSError):
                sock.close()
        for path in (self.command_path, self.health_path):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)


@dataclass
class _TargetSession:
    session_id: int
    worker_id: int
    source_rank: int
    pid: int
    runtime_timeout_s: float
    sockets: dict[Lane, socket.socket]
    locks: dict[Lane, threading.Lock] = field(
        default_factory=lambda: {Lane.COMMAND: threading.Lock(), Lane.HEALTH: threading.Lock()}
    )

    def close(self) -> None:
        for lane, sock in self.sockets.items():
            with self.locks[lane]:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
                with contextlib.suppress(OSError):
                    sock.close()


class SidecarProxy:
    def __init__(
        self,
        *,
        rank: int,
        sidecar_socket: str,
        session_dir: str,
        worker_ids: set[int],
        bootstrap_socket: str | None,
    ) -> None:
        self.rank = rank
        self.sidecar_socket = sidecar_socket
        self.session_dir = session_dir
        self.worker_ids = worker_ids
        self.bootstrap_socket = bootstrap_socket
        self._sidecar: socket.socket | None = None
        self._sidecar_send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._pending: dict[int, _PendingOpen] = {}
        self._sources: dict[int, _SourceSession] = {}
        self._opening_source_sessions: set[int] = set()
        self._targets: dict[int, _TargetSession] = {}
        self._closed_sessions: set[int] = set()
        self._listeners: list[socket.socket] = []

    def _send(self, envelope: Envelope) -> None:
        sidecar = self._sidecar
        if sidecar is None:
            raise RuntimeError("MPI sidecar is not connected")
        data = encode_envelope(envelope)
        with self._sidecar_send_lock:
            sidecar.sendall(data)

    def _send_message(
        self,
        message_type: MessageType,
        *,
        target_rank: int,
        session_id: int = 0,
        lane: Lane = Lane.BOOTSTRAP,
        sequence: int = 0,
        payload: bytes = b"",
    ) -> None:
        self._send(Envelope(message_type, -1, target_rank, session_id, lane, sequence, payload))

    def _session_paths(self, session_id: int) -> tuple[str, str]:
        root = Path(self.session_dir) / f"s{session_id:x}"
        return str(root / "cmd.sock"), str(root / "health.sock")

    def _accept_source_lane(self, session: _SourceSession, lane: Lane, listener: socket.socket) -> None:
        try:
            conn, _ = listener.accept()
            with session.lock:
                session.sockets[lane] = conn
                queued = session.queued[lane]
                session.queued[lane] = []
            for frame in queued:
                conn.sendall(frame)
            while not self._stop.is_set():
                frame = _read_slr3(conn)
                frame_session, worker_id, sequence = _slr3_identity(frame)
                if frame_session != session.session_id or worker_id != session.worker_id:
                    raise ValueError("local L4 SLR3 identity differs from sidecar session")
                self._send_message(
                    MessageType.FRAME,
                    target_rank=session.target_rank,
                    session_id=session.session_id,
                    lane=lane,
                    sequence=sequence,
                    payload=frame,
                )
                _log_frame(
                    "L4_TO_MPI",
                    Envelope(
                        MessageType.FRAME,
                        self.rank,
                        session.target_rank,
                        session.session_id,
                        lane,
                        sequence,
                        frame,
                    ),
                )
        except (EOFError, OSError, ValueError):
            pass
        finally:
            if lane == Lane.COMMAND and not self._stop.is_set():
                with contextlib.suppress(OSError, RuntimeError):
                    self._send_message(
                        MessageType.CLOSE_SESSION,
                        target_rank=session.target_rank,
                        session_id=session.session_id,
                    )
                self._close_session(session.session_id)

    def _create_source_session(self, session_id: int, worker_id: int, target_rank: int) -> _SourceSession:
        if self._stop.is_set():
            raise RuntimeError("MPI sidecar is stopping")
        with self._state_lock:
            if session_id in self._sources or session_id in self._opening_source_sessions:
                raise RuntimeError("duplicate source session id")
            self._opening_source_sessions.add(session_id)
        try:
            command_path, health_path = self._session_paths(session_id)
            command_listener = _bind_unix(command_path)
            try:
                health_listener = _bind_unix(health_path)
            except BaseException:
                command_listener.close()
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(command_path)
                raise
            session = _SourceSession(
                session_id,
                worker_id,
                target_rank,
                command_path,
                health_path,
                command_listener,
                health_listener,
            )
            with self._state_lock:
                if self._stop.is_set():
                    session.close()
                    raise RuntimeError("MPI sidecar stopped during source session creation")
                self._closed_sessions.discard(session_id)
                self._sources[session_id] = session
        finally:
            with self._state_lock:
                self._opening_source_sessions.discard(session_id)
        threading.Thread(
            target=self._accept_source_lane, args=(session, Lane.COMMAND, command_listener), daemon=True
        ).start()
        threading.Thread(
            target=self._accept_source_lane, args=(session, Lane.HEALTH, health_listener), daemon=True
        ).start()
        return session

    def _handle_bootstrap(self, conn: socket.socket) -> None:
        pending: _PendingOpen | None = None
        session_id = 0
        try:
            request = _read_json(conn)
            if request.get("op") == "STOP_WORLD":
                self._send_message(MessageType.SHUTDOWN, target_rank=self.rank)
                _send_json(conn, {"ok": True})
                return
            if request.get("version") != 1 or request.get("op") != "OPEN_SESSION":
                raise ValueError("unsupported sidecar bootstrap request")
            target_rank = int(request["target_rank"])
            manifest = request.get("manifest")
            if not isinstance(manifest, dict):
                raise ValueError("OPEN_SESSION manifest must be an object")
            session_id = int(manifest["session_id"])
            if session_id == 0:
                raise ValueError("session_id must be non-zero")
            remaining = float(manifest["startup_remaining_s"])
            if remaining <= 0:
                raise TimeoutError("OPEN_SESSION startup budget is exhausted")
            started = time.monotonic()
            pending = _PendingOpen()
            with self._state_lock:
                if session_id in self._pending:
                    raise RuntimeError("duplicate pending session id")
                self._pending[session_id] = pending
            forwarded = dict(request)
            forwarded_manifest = dict(manifest)
            forwarded_manifest["startup_remaining_s"] = max(0.0, remaining - (time.monotonic() - started))
            forwarded["manifest"] = forwarded_manifest
            payload = json.dumps(forwarded, sort_keys=True).encode("utf-8")
            self._send_message(
                MessageType.OPEN_SESSION,
                target_rank=target_rank,
                session_id=session_id,
                payload=payload,
            )
            if not pending.event.wait(timeout=remaining):
                raise TimeoutError("MPI OPEN_SESSION reply timed out")
            assert pending.reply is not None
            _send_json(conn, pending.reply)
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(OSError):
                _send_json(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if pending is not None:
                with self._state_lock:
                    self._pending.pop(session_id, None)

    def _bootstrap_loop(self, listener: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve_bootstrap_connection, args=(conn,), daemon=True).start()

    def _serve_bootstrap_connection(self, conn: socket.socket) -> None:
        with conn:
            self._handle_bootstrap(conn)

    def _connect_runner(self, host: str, port: int, timeout: float) -> socket.socket:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(None)
        return sock

    def _forward_target_lane(self, session: _TargetSession, lane: Lane) -> None:
        try:
            sock = session.sockets[lane]
            while not self._stop.is_set():
                frame = _read_slr3(sock)
                frame_session, worker_id, sequence = _slr3_identity(frame)
                if frame_session != session.session_id or worker_id != session.worker_id:
                    raise ValueError("remote L3 SLR3 identity differs from sidecar session")
                self._send_message(
                    MessageType.FRAME,
                    target_rank=session.source_rank,
                    session_id=session.session_id,
                    lane=lane,
                    sequence=sequence,
                    payload=frame,
                )
                _log_frame(
                    "L3_TO_MPI",
                    Envelope(
                        MessageType.FRAME,
                        self.rank,
                        session.source_rank,
                        session.session_id,
                        lane,
                        sequence,
                        frame,
                    ),
                )
        except (EOFError, OSError, ValueError) as exc:
            payload = json.dumps({"error": f"{type(exc).__name__}: {exc}"}, sort_keys=True).encode("utf-8")
            with contextlib.suppress(OSError, RuntimeError):
                self._send_message(
                    MessageType.CLOSE_SESSION,
                    target_rank=session.source_rank,
                    session_id=session.session_id,
                    lane=lane,
                    payload=payload,
                )
            if lane == Lane.COMMAND:
                self._close_session(session.session_id)

    def _open_target(self, envelope: Envelope) -> None:
        session: _TargetSession | None = None
        session_registered = False
        try:
            request = json.loads(envelope.payload.decode("utf-8"))
            manifest = request["manifest"]
            if not isinstance(manifest, dict):
                raise ValueError("OPEN_SESSION manifest must be an object")
            worker_id = int(manifest["worker_id"])
            if worker_id not in self.worker_ids:
                raise ValueError(f"worker_id {worker_id} is not assigned to MPI rank {self.rank}")
            if int(manifest["session_id"]) != envelope.session_id:
                raise ValueError("OPEN_SESSION envelope and manifest session differ")
            timeout = float(manifest["startup_remaining_s"])
            if timeout <= 0:
                raise TimeoutError("target startup budget is exhausted")
            daemon_host = str(request["daemon_host"])
            daemon_port = int(request["daemon_port"])
            started = time.monotonic()
            daemon = socket.create_connection((daemon_host, daemon_port), timeout=timeout)
            with daemon:
                forwarded = dict(manifest)
                forwarded["startup_remaining_s"] = max(0.0, timeout - (time.monotonic() - started))
                _send_json(daemon, forwarded)
                reply = _read_json(daemon)
            if not reply.get("ok", False):
                raise RuntimeError(f"remote daemon rejected session: {reply.get('error')}")
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("target startup budget exhausted before runner attach")
            command = self._connect_runner(str(reply["command_host"]), int(reply["command_port"]), remaining)
            try:
                health = self._connect_runner(str(reply["health_host"]), int(reply["health_port"]), remaining)
            except BaseException:
                command.close()
                raise
            session = _TargetSession(
                envelope.session_id,
                worker_id,
                envelope.source_rank,
                int(reply.get("pid", 0)),
                float(manifest["session_timeout_s"]),
                {Lane.COMMAND: command, Lane.HEALTH: health},
            )
            with self._state_lock:
                if self._stop.is_set():
                    session.close()
                    session = None
                    raise RuntimeError("MPI sidecar stopped during target session creation")
                if envelope.session_id in self._targets:
                    session.close()
                    session = None
                    raise RuntimeError("duplicate target session id")
                self._closed_sessions.discard(envelope.session_id)
                self._targets[envelope.session_id] = session
                session_registered = True
            assert session is not None
            result = {"ok": True, "worker_id": worker_id, "pid": session.pid}
            self._send_message(
                MessageType.OPEN_SESSION_REPLY,
                target_rank=envelope.source_rank,
                session_id=envelope.session_id,
                payload=json.dumps(result, sort_keys=True).encode("utf-8"),
            )
            for lane in (Lane.COMMAND, Lane.HEALTH):
                threading.Thread(target=self._forward_target_lane, args=(session, lane), daemon=True).start()
        except Exception as exc:  # noqa: BLE001
            if session_registered:
                self._close_session(envelope.session_id)
            elif session is not None:
                session.close()
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            with contextlib.suppress(OSError, RuntimeError):
                self._send_message(
                    MessageType.OPEN_SESSION_REPLY,
                    target_rank=envelope.source_rank,
                    session_id=envelope.session_id,
                    payload=json.dumps(result, sort_keys=True).encode("utf-8"),
                )

    def _handle_open_reply(self, envelope: Envelope) -> None:
        reply = json.loads(envelope.payload.decode("utf-8"))
        with self._state_lock:
            pending = self._pending.get(envelope.session_id)
        if pending is None:
            self._send_message(
                MessageType.CLOSE_SESSION,
                target_rank=envelope.source_rank,
                session_id=envelope.session_id,
            )
            return
        if reply.get("ok", False):
            try:
                worker_id = int(reply["worker_id"])
                session = self._create_source_session(envelope.session_id, worker_id, envelope.source_rank)
                reply = {
                    "ok": True,
                    "command_path": session.command_path,
                    "health_path": session.health_path,
                    "pid": int(reply.get("pid", 0)),
                }
            except Exception as exc:  # noqa: BLE001
                with contextlib.suppress(OSError, RuntimeError):
                    self._send_message(
                        MessageType.CLOSE_SESSION,
                        target_rank=envelope.source_rank,
                        session_id=envelope.session_id,
                    )
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        pending.reply = reply
        pending.event.set()

    def _deliver_frame(self, envelope: Envelope) -> None:
        frame_session, worker_id, sequence = _slr3_identity(envelope.payload)
        if frame_session != envelope.session_id or sequence != envelope.sequence:
            raise ValueError("sidecar envelope and SLR3 frame identity differ")
        with self._state_lock:
            source = self._sources.get(envelope.session_id)
            target = self._targets.get(envelope.session_id)
            closed = envelope.session_id in self._closed_sessions
        if source is not None and envelope.source_rank == source.target_rank:
            if worker_id != source.worker_id:
                raise ValueError("source session worker_id mismatch")
            _log_frame("MPI_TO_L4", envelope)
            with source.lock:
                sock = source.sockets.get(envelope.lane)
                if sock is None:
                    source.queued[envelope.lane].append(envelope.payload)
                    return
                sock.sendall(envelope.payload)
            return
        if target is not None and envelope.source_rank == target.source_rank:
            if worker_id != target.worker_id:
                raise ValueError("target session worker_id mismatch")
            _log_frame("MPI_TO_L3", envelope)
            sock = target.sockets[envelope.lane]
            with target.locks[envelope.lane]:
                try:
                    sock.settimeout(target.runtime_timeout_s)
                    sock.sendall(envelope.payload)
                finally:
                    sock.settimeout(None)
            return
        if closed:
            return
        raise ValueError("FRAME does not match a local sidecar session")

    @staticmethod
    def _wait_remote_runner_exit(pid: int, timeout_s: float) -> bool:
        if pid <= 0:
            return True
        deadline = time.monotonic() + min(timeout_s, 5.0)
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
            time.sleep(0.02)
        return False

    def _close_session(self, session_id: int, *, require_runner_exit: bool = False) -> None:
        with self._state_lock:
            source = self._sources.pop(session_id, None)
            target = self._targets.pop(session_id, None)
            self._closed_sessions.add(session_id)
        if source is not None:
            source.close()
        if target is not None:
            target.close()
            exited = self._wait_remote_runner_exit(target.pid, target.runtime_timeout_s)
            print(
                json.dumps(
                    {
                        "event": "REMOTE_L3_SESSION_CLOSED",
                        "rank": self.rank,
                        "session_id": session_id,
                        "pid": target.pid,
                        "runner_exited": exited,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if require_runner_exit and not exited:
                raise RuntimeError(f"remote L3 runner pid {target.pid} did not exit within cleanup deadline")

    def _sidecar_loop(self) -> None:
        assert self._sidecar is not None
        while not self._stop.is_set():
            envelope = read_envelope(self._sidecar)
            if envelope.target_rank != self.rank:
                raise ValueError("sidecar delivered an envelope for another rank")
            if envelope.message_type == MessageType.WORLD_READY:
                world = json.loads(envelope.payload.decode("utf-8"))
                if int(world["rank"]) != self.rank:
                    raise ValueError("MPI world rank differs from proxy rank")
                mapped_workers = _worker_ids_for_rank(str(world["worker_map"]), self.rank)
                if mapped_workers != self.worker_ids:
                    raise ValueError("MPI worker map differs from proxy worker assignment")
                print(json.dumps(world, sort_keys=True), flush=True)
                if self.rank == 0:
                    if self.bootstrap_socket is None:
                        raise ValueError("rank 0 proxy requires --bootstrap-socket")
                    listener = _bind_unix(self.bootstrap_socket)
                    self._listeners.append(listener)
                    threading.Thread(target=self._bootstrap_loop, args=(listener,), daemon=True).start()
                    print(f"L4 bootstrap READY {self.bootstrap_socket}", flush=True)
            elif envelope.message_type == MessageType.OPEN_SESSION:
                threading.Thread(target=self._open_target, args=(envelope,), daemon=True).start()
            elif envelope.message_type == MessageType.OPEN_SESSION_REPLY:
                self._handle_open_reply(envelope)
            elif envelope.message_type == MessageType.FRAME:
                self._deliver_frame(envelope)
            elif envelope.message_type == MessageType.CLOSE_SESSION:
                self._close_session(envelope.session_id, require_runner_exit=True)
            elif envelope.message_type == MessageType.ERROR:
                raise RuntimeError(envelope.payload.decode("utf-8", errors="replace"))
            elif envelope.message_type == MessageType.SHUTDOWN:
                self._stop.set()

    def serve(self) -> int:
        listener = _bind_unix(self.sidecar_socket)
        self._listeners.append(listener)
        print(f"MPI sidecar proxy rank={self.rank} LISTENING {self.sidecar_socket}", flush=True)
        try:
            self._sidecar, _ = listener.accept()
            self._sidecar_loop()
            return 0
        finally:
            self._stop.set()
            for pending in self._pending.values():
                pending.reply = {"ok": False, "error": "MPI sidecar stopped"}
                pending.event.set()
            for session_id in list(self._sources) + list(self._targets):
                self._close_session(session_id)
            for sock in self._listeners:
                with contextlib.suppress(OSError):
                    sock.close()
            if self._sidecar is not None:
                with contextlib.suppress(OSError):
                    self._sidecar.close()
            for path in (self.sidecar_socket, self.bootstrap_socket):
                if path:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(path)


def stop_world(bootstrap_socket: str, timeout_s: float = 5.0) -> int:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(bootstrap_socket)
        _send_json(sock, {"version": 1, "op": "STOP_WORLD"})
        reply = _read_json(sock)
    if not reply.get("ok", False):
        raise RuntimeError(f"MPI sidecar stop failed: {reply.get('error')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int)
    parser.add_argument("--sidecar-socket")
    parser.add_argument("--session-dir")
    parser.add_argument("--worker-ids", default="")
    parser.add_argument("--bootstrap-socket")
    parser.add_argument("--stop-world")
    ns = parser.parse_args(argv)
    if ns.stop_world:
        return stop_world(ns.stop_world)
    if ns.rank is None or not ns.sidecar_socket or not ns.session_dir:
        parser.error("--rank, --sidecar-socket and --session-dir are required in server mode")
    worker_ids = {int(item) for item in ns.worker_ids.split(",") if item}
    proxy = SidecarProxy(
        rank=ns.rank,
        sidecar_socket=ns.sidecar_socket,
        session_dir=ns.session_dir,
        worker_ids=worker_ids,
        bootstrap_socket=ns.bootstrap_socket,
    )
    return proxy.serve()


if __name__ == "__main__":
    sys.exit(main())
