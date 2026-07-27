# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Remote L3 buffer allocation for real L2 children and existing sim workers."""

from __future__ import annotations

import ctypes

import pytest

from simpler.remote_l3_session import _allocate_remote_buffer


class _FakeHostBuffer:
    def __init__(self, nbytes: int):
        self._backing = (ctypes.c_ubyte * nbytes)()
        self.data_ptr = ctypes.addressof(self._backing)
        self.buffer = memoryview(self._backing)


class _FakeInnerWorker:
    def __init__(self, error: str | None = None):
        self.error = error
        self.freed: list[_FakeHostBuffer] = []

    def create_host_buffer(self, nbytes: int) -> _FakeHostBuffer:
        if self.error is not None:
            raise RuntimeError(self.error)
        return _FakeHostBuffer(nbytes)

    def free_host_buffer(self, handle: _FakeHostBuffer) -> None:
        self.freed.append(handle)


def test_remote_buffer_uses_inner_worker_host_buffer() -> None:
    worker = _FakeInnerWorker()

    entry = _allocate_remote_buffer(worker, 64, "a2a3")  # type: ignore[arg-type]

    assert entry.addr == entry.data.data_ptr
    entry.close(unlink=True)
    assert worker.freed == [entry.data]


def test_remote_buffer_falls_back_for_childless_worker() -> None:
    worker = _FakeInnerWorker("create_host_buffer requires at least one forked chip or sub child")

    entry = _allocate_remote_buffer(worker, 64, "a2a3")  # type: ignore[arg-type]
    try:
        assert entry.shm_name
        assert entry.addr != 0
    finally:
        entry.close(unlink=True)


def test_remote_buffer_does_not_hide_other_allocation_failures() -> None:
    worker = _FakeInnerWorker("MAP_HOST failed")

    with pytest.raises(RuntimeError, match="MAP_HOST failed"):
        _allocate_remote_buffer(worker, 64, "a2a3")  # type: ignore[arg-type]


def test_sim_remote_buffer_keeps_shared_memory_backing() -> None:
    worker = _FakeInnerWorker("create_host_buffer must not be called")

    entry = _allocate_remote_buffer(worker, 64, "a2a3sim")  # type: ignore[arg-type]
    try:
        assert entry.shm_name
        assert entry.owner is None
    finally:
        entry.close(unlink=True)
