# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rank-0 UDS gateway and opaque MPI envelope routing for direct MPI L3."""

from __future__ import annotations

import contextlib
import enum
import hashlib
import json
import os
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from .remote_l3_protocol import FrameHeader, FrameType, encode_frame
from .remote_l3_session import _run_command_loop
from .worker import Worker

MPI_L3_MAGIC = b"ML3P"
MPI_L3_VERSION = 1
MPI_L3_HEADER = struct.Struct("<4sIIiiQIIQ")
MPI_L3_MAX_PAYLOAD = 16 * 1024 * 1024
SLR3_HEADER_BYTES = 40
SLR3_MAX_PAYLOAD = 16 * 1024 * 1024


class MessageType(enum.IntEnum):
    OPEN_SESSION = 1
    OPEN_SESSION_REPLY = 2
    CLOSE_SESSION = 3
    SHUTDOWN = 4
    FRAME_L4_TO_L3 = 5
    FRAME_L3_TO_L4 = 6


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


class MpiTransport(Protocol):
    def send(self, target_rank: int, payload: bytes) -> None: ...

    def receive(self) -> tuple[int, bytes] | None: ...


def encode_envelope(envelope: Envelope) -> bytes:
    payload = bytes(envelope.payload)
    if len(payload) > MPI_L3_MAX_PAYLOAD:
        raise ValueError("MPI L3 payload exceeds maximum")
    return MPI_L3_HEADER.pack(
        MPI_L3_MAGIC,
        MPI_L3_VERSION,
        int(envelope.message_type),
        int(envelope.source_rank),
        int(envelope.target_rank),
        int(envelope.session_id),
        int(envelope.lane),
        len(payload),
        int(envelope.sequence),
    ) + payload


def decode_envelope(data: bytes) -> Envelope:
    if len(data) < MPI_L3_HEADER.size:
        raise ValueError("MPI L3 envelope is truncated")
    magic, version, raw_type, source, target, session_id, raw_lane, payload_size, sequence = (
        MPI_L3_HEADER.unpack_from(data)
    )
    if magic != MPI_L3_MAGIC or version != MPI_L3_VERSION:
        raise ValueError("MPI L3 envelope magic or version mismatch")
    if payload_size > MPI_L3_MAX_PAYLOAD or len(data) != MPI_L3_HEADER.size + payload_size:
        raise ValueError("MPI L3 envelope payload length is invalid")
    try:
        message_type = MessageType(raw_type)
        lane = Lane(raw_lane)
    except ValueError as exc:
        raise ValueError("MPI L3 envelope type or lane is unknown") from exc
    return Envelope(message_type, source, target, session_id, lane, sequence, data[MPI_L3_HEADER.size :])


def _read_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("socket closed")
        data.extend(chunk)
    return bytes(data)


def _send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    if len(data) > MPI_L3_MAX_PAYLOAD:
        raise ValueError("JSON payload exceeds maximum")
    sock.sendall(struct.pack("<I", len(data)) + data)


def _read_json(sock: socket.socket) -> dict[str, Any]:
    size = struct.unpack("<I", _read_exact(sock, 4))[0]
    if size > MPI_L3_MAX_PAYLOAD:
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


def _log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _log_frame(event: str, envelope: Envelope) -> None:
    _session_id, worker_id, sequence = _slr3_identity(envelope.payload)
    frame_type = struct.unpack_from("<I", envelope.payload, 8)[0]
    _log_event(
        event,
        rank=envelope.source_rank,
        target_rank=envelope.target_rank,
        session_id=envelope.session_id,
        worker_id=worker_id,
        lane=envelope.lane.name,
        frame_type=frame_type,
        sequence=sequence,
        sha256=hashlib.sha256(envelope.payload).hexdigest(),
    )


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


class MpiL3Gateway:  # noqa: PLR0904 -- owns bootstrap, source, target, routing, and lifecycle boundaries
    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        bootstrap_socket: str | None,
        session_dir: str,
        inner_worker: Worker,
        rank_config: dict[str, Any],
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.bootstrap_socket = bootstrap_socket
        self.session_dir = session_dir
        self.inner_worker = inner_worker
        self.rank_config = rank_config
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._pending: dict[int, _PendingOpen] = {}
        self._sources: dict[int, _SourceSession] = {}
        self._opening_source_sessions: set[int] = set()
        self._targets: dict[int, _TargetSession] = {}
        self._closed_sessions: set[int] = set()
        self._listeners: list[socket.socket] = []
        self._outbound: queue.Queue[Envelope] = queue.Queue()
        self._direct_resources: dict[int, tuple[threading.Event, list[socket.socket], list[threading.Thread]]] = {}

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
        self._outbound.put(Envelope(message_type, -1, target_rank, session_id, lane, sequence, payload))

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
                    raise ValueError("local L4 SLR3 identity differs from MPI L3 session")
                envelope = Envelope(
                    MessageType.FRAME_L4_TO_L3,
                    self.rank,
                    session.target_rank,
                    session.session_id,
                    lane,
                    sequence,
                    frame,
                )
                self._send_message(
                    MessageType.FRAME_L4_TO_L3,
                    target_rank=session.target_rank,
                    session_id=session.session_id,
                    lane=lane,
                    sequence=sequence,
                    payload=frame,
                )
                _log_frame("L4_TO_MPI", envelope)
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
            raise RuntimeError("MPI L3 gateway is stopping")
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
                    raise RuntimeError("MPI L3 gateway stopped during source session creation")
                self._closed_sessions.discard(session_id)
                self._sources[session_id] = session
        finally:
            with self._state_lock:
                self._opening_source_sessions.discard(session_id)
        threading.Thread(
            target=self._accept_source_lane,
            args=(session, Lane.COMMAND, command_listener),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._accept_source_lane,
            args=(session, Lane.HEALTH, health_listener),
            daemon=True,
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
                raise ValueError("unsupported MPI L3 gateway bootstrap request")
            target_rank = int(request["target_rank"])
            if target_rank < 0 or target_rank >= self.world_size:
                raise ValueError("OPEN_SESSION target_rank is outside the MPI world")
            manifest = request.get("manifest")
            if not isinstance(manifest, dict):
                raise ValueError("OPEN_SESSION manifest must be an object")
            session_id = int(manifest["session_id"])
            if session_id == 0:
                raise ValueError("session_id must be non-zero")
            remaining = float(manifest["startup_remaining_s"])
            if remaining <= 0:
                raise TimeoutError("OPEN_SESSION startup budget is exhausted")
            _log_event(
                "L4_OPEN_SESSION_MPI_L3",
                rank=self.rank,
                target_rank=target_rank,
                session_id=session_id,
                worker_id=int(manifest["worker_id"]),
                local_transport="unix",
                cross_machine_transport="mpi",
            )
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
            self._send_message(
                MessageType.OPEN_SESSION,
                target_rank=target_rank,
                session_id=session_id,
                payload=json.dumps(forwarded, sort_keys=True).encode("utf-8"),
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

    def _forward_target_lane(self, session: _TargetSession, lane: Lane) -> None:
        try:
            sock = session.sockets[lane]
            while not self._stop.is_set():
                frame = _read_slr3(sock)
                frame_session, worker_id, sequence = _slr3_identity(frame)
                if frame_session != session.session_id or worker_id != session.worker_id:
                    raise ValueError("local L3 SLR3 identity differs from MPI L3 session")
                envelope = Envelope(
                    MessageType.FRAME_L3_TO_L4,
                    self.rank,
                    session.source_rank,
                    session.session_id,
                    lane,
                    sequence,
                    frame,
                )
                self._send_message(
                    MessageType.FRAME_L3_TO_L4,
                    target_rank=session.source_rank,
                    session_id=session.session_id,
                    lane=lane,
                    sequence=sequence,
                    payload=frame,
                )
                _log_frame("L3_TO_MPI", envelope)
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

    @staticmethod
    def _health_producer(
        sock: socket.socket, stop: threading.Event, session_id: int, worker_id: int
    ) -> None:
        sequence = 0
        try:
            while not stop.wait(0.2):
                sequence += 1
                sock.sendall(encode_frame(FrameHeader(FrameType.HEALTH, session_id, worker_id, sequence)))
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                sock.close()

    def _command_executor(
        self, sock: socket.socket, manifest: dict[str, Any], stop: threading.Event
    ) -> None:
        try:
            _run_command_loop(sock, manifest, self.inner_worker)
        finally:
            stop.set()
            with contextlib.suppress(OSError):
                sock.close()

    def _validate_manifest(self, manifest: dict[str, Any], envelope: Envelope) -> int:
        worker_id = int(manifest["worker_id"])
        expected_worker_id = int(self.rank_config.get("worker_id", self.rank))
        if worker_id != expected_worker_id:
            raise ValueError(f"worker_id {worker_id} is not assigned to MPI rank {self.rank}")
        if int(manifest["session_id"]) != envelope.session_id:
            raise ValueError("OPEN_SESSION envelope and manifest session differ")
        expected = {
            "platform": str(self.rank_config["platform"]),
            "runtime": str(self.rank_config["runtime"]),
            "device_ids": [int(item) for item in self.rank_config["device_ids"]],
            "num_sub_workers": int(self.rank_config.get("num_sub_workers", 0)),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"OPEN_SESSION {key} differs from the pre-launched MPI L3 rank")
        if float(manifest["startup_remaining_s"]) <= 0:
            raise TimeoutError("target startup budget is exhausted")
        return worker_id

    def _open_target(self, envelope: Envelope) -> None:  # noqa: PLR0915 -- atomic direct-session construction
        session: _TargetSession | None = None
        registered = False
        resources: tuple[threading.Event, list[socket.socket], list[threading.Thread]] | None = None
        try:
            request = json.loads(envelope.payload.decode("utf-8"))
            if request.get("version") != 1 or request.get("op") != "OPEN_SESSION":
                raise ValueError("unsupported direct MPI L3 OPEN_SESSION request")
            if int(request["target_rank"]) != self.rank:
                raise ValueError("OPEN_SESSION request target differs from MPI envelope")
            manifest = request["manifest"]
            if not isinstance(manifest, dict):
                raise ValueError("OPEN_SESSION manifest must be an object")
            worker_id = self._validate_manifest(manifest, envelope)
            with self._state_lock:
                if self._targets:
                    raise RuntimeError("direct MPI L3 rank supports one active L4 session")
            command_router, command_worker = socket.socketpair()
            health_router, health_worker = socket.socketpair()
            stop = threading.Event()
            threads: list[threading.Thread] = []
            resources = (stop, [command_worker, health_worker], threads)
            session = _TargetSession(
                envelope.session_id,
                worker_id,
                envelope.source_rank,
                float(manifest["session_timeout_s"]),
                {Lane.COMMAND: command_router, Lane.HEALTH: health_router},
            )
            with self._state_lock:
                if self._stop.is_set() or envelope.session_id in self._targets:
                    raise RuntimeError("direct MPI L3 stopped or received a duplicate session")
                self._closed_sessions.discard(envelope.session_id)
                self._targets[envelope.session_id] = session
                registered = True
            threads.extend(
                [
                    threading.Thread(
                        target=self._forward_target_lane,
                        args=(session, Lane.COMMAND),
                        daemon=True,
                    ),
                    threading.Thread(
                        target=self._forward_target_lane,
                        args=(session, Lane.HEALTH),
                        daemon=True,
                    ),
                    threading.Thread(
                        target=self._health_producer,
                        args=(health_worker, stop, envelope.session_id, worker_id),
                        daemon=True,
                    ),
                    threading.Thread(
                        target=self._command_executor,
                        args=(command_worker, manifest, stop),
                        daemon=True,
                    ),
                ]
            )
            self._direct_resources[envelope.session_id] = resources
            result = {"ok": True, "worker_id": worker_id, "pid": os.getpid()}
            self._send_message(
                MessageType.OPEN_SESSION_REPLY,
                target_rank=envelope.source_rank,
                session_id=envelope.session_id,
                payload=json.dumps(result, sort_keys=True).encode("utf-8"),
            )
            _log_event(
                "MPI_L3_SESSION_READY",
                rank=self.rank,
                source_rank=envelope.source_rank,
                session_id=envelope.session_id,
                worker_id=worker_id,
                l3_pid=os.getpid(),
                l4_to_l3="uds+mpi",
                l3_to_l2="mailbox",
            )
            for thread in threads:
                thread.start()
        except Exception as exc:  # noqa: BLE001
            if registered:
                self._close_session(envelope.session_id)
            elif session is not None:
                session.close()
            if resources is not None:
                resources[0].set()
                for sock in resources[1]:
                    with contextlib.suppress(OSError):
                        sock.close()
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
                _log_event(
                    "L4_MPI_L3_UDS_READY",
                    rank=self.rank,
                    target_rank=envelope.source_rank,
                    session_id=envelope.session_id,
                    worker_id=worker_id,
                    command_path=session.command_path,
                    health_path=session.health_path,
                )
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
            raise ValueError("MPI envelope and SLR3 frame identity differ")
        with self._state_lock:
            source = self._sources.get(envelope.session_id)
            target = self._targets.get(envelope.session_id)
            closed = envelope.session_id in self._closed_sessions
        source_matches = source is not None and envelope.source_rank == source.target_rank
        target_matches = target is not None and envelope.source_rank == target.source_rank
        if envelope.message_type == MessageType.FRAME_L3_TO_L4 and source_matches:
            assert source is not None
            if worker_id != source.worker_id:
                raise ValueError("source session worker_id mismatch")
            _log_frame("MPI_TO_L4", envelope)
            with source.lock:
                sock = source.sockets.get(envelope.lane)
                if sock is None:
                    source.queued[envelope.lane].append(envelope.payload)
                else:
                    sock.sendall(envelope.payload)
            return
        if envelope.message_type == MessageType.FRAME_L4_TO_L3 and target_matches:
            assert target is not None
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
        raise ValueError("FRAME does not match a local MPI L3 session")

    def _close_session(self, session_id: int) -> None:
        resources = self._direct_resources.pop(session_id, None)
        with self._state_lock:
            source = self._sources.pop(session_id, None)
            target = self._targets.pop(session_id, None)
            self._closed_sessions.add(session_id)
        if source is not None:
            source.close()
        if target is not None:
            target.close()
        if resources is None:
            return
        stop, sockets, threads = resources
        stop.set()
        for sock in sockets:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                sock.close()
        current = threading.current_thread()
        for thread in threads:
            if thread is not current and thread.is_alive():
                thread.join(timeout=1.0)

    def _dispatch(self, envelope: Envelope) -> None:
        if envelope.target_rank != self.rank:
            raise ValueError("MPI envelope target rank differs from local rank")
        if envelope.message_type == MessageType.OPEN_SESSION:
            threading.Thread(target=self._open_target, args=(envelope,), daemon=True).start()
        elif envelope.message_type == MessageType.OPEN_SESSION_REPLY:
            self._handle_open_reply(envelope)
        elif envelope.message_type in (MessageType.FRAME_L4_TO_L3, MessageType.FRAME_L3_TO_L4):
            self._deliver_frame(envelope)
        elif envelope.message_type == MessageType.CLOSE_SESSION:
            self._close_session(envelope.session_id)
        elif envelope.message_type == MessageType.SHUTDOWN:
            self._stop.set()
        else:
            raise ValueError(f"unsupported direct MPI L3 envelope {envelope.message_type.name}")

    def _route_outbound(self, transport: MpiTransport, envelope: Envelope) -> None:
        envelope = replace(envelope, source_rank=self.rank)
        if envelope.message_type == MessageType.SHUTDOWN:
            if self.rank != 0:
                raise RuntimeError("only MPI rank 0 may stop the world")
            for target in range(self.world_size):
                routed = replace(envelope, target_rank=target)
                if target == self.rank:
                    self._dispatch(routed)
                else:
                    transport.send(target, encode_envelope(routed))
            return
        if envelope.target_rank == self.rank:
            if envelope.lane != Lane.HEALTH:
                _log_event(
                    "MPI_L3_ROUTE_LOCAL",
                    rank=self.rank,
                    message_type=envelope.message_type.name,
                    session_id=envelope.session_id,
                )
            self._dispatch(envelope)
        else:
            if envelope.lane != Lane.HEALTH:
                _log_event(
                    "MPI_L3_SEND",
                    rank=self.rank,
                    target_rank=envelope.target_rank,
                    message_type=envelope.message_type.name,
                    session_id=envelope.session_id,
                )
            transport.send(envelope.target_rank, encode_envelope(envelope))

    def serve(self, transport: MpiTransport, world_record: dict[str, Any]) -> int:
        if self.rank == 0:
            if self.bootstrap_socket is None:
                raise ValueError("MPI rank 0 requires a bootstrap socket")
            listener = _bind_unix(self.bootstrap_socket)
            self._listeners.append(listener)
            threading.Thread(target=self._bootstrap_loop, args=(listener,), daemon=True).start()
            _log_event("L4_MPI_L3_GATEWAY_READY", path=self.bootstrap_socket, transport="unix")
        print(json.dumps(world_record, sort_keys=True), flush=True)
        try:
            while not self._stop.is_set():
                for _ in range(64):
                    try:
                        envelope = self._outbound.get_nowait()
                    except queue.Empty:
                        break
                    self._route_outbound(transport, envelope)
                while True:
                    received = transport.receive()
                    if received is None:
                        break
                    source, payload = received
                    envelope = decode_envelope(payload)
                    if envelope.source_rank != source:
                        raise ValueError("MPI source rank differs from envelope metadata")
                    if envelope.lane != Lane.HEALTH:
                        _log_event(
                            "MPI_L3_RECV",
                            rank=self.rank,
                            source_rank=source,
                            message_type=envelope.message_type.name,
                            session_id=envelope.session_id,
                        )
                    self._dispatch(envelope)
                time.sleep(0.005)
            return 0
        finally:
            self._stop.set()
            for pending in self._pending.values():
                pending.reply = {"ok": False, "error": "direct MPI L3 stopped"}
                pending.event.set()
            for session_id in list(self._sources) + list(self._targets):
                self._close_session(session_id)
            for listener in self._listeners:
                with contextlib.suppress(OSError):
                    listener.close()
            if self.bootstrap_socket:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(self.bootstrap_socket)


def stop_world(bootstrap_socket: str, timeout_s: float = 5.0) -> int:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(bootstrap_socket)
        _send_json(sock, {"version": 1, "op": "STOP_WORLD"})
        reply = _read_json(sock)
    if not reply.get("ok", False):
        raise RuntimeError(f"MPI world stop failed: {reply.get('error')}")
    return 0
