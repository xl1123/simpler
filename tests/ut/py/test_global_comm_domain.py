# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections import Counter

import pytest
from simpler.global_comm_domain import (
    GLOBAL_DOMAIN_DESCRIPTOR_BYTES,
    GLOBAL_DOMAIN_PROFILE_IDS,
    GLOBAL_DOMAIN_VERSION,
    GlobalCommInitCommand,
    GlobalDomainBuffer,
    GlobalDomainCommand,
    GlobalDomainDescriptor,
    GlobalDomainMember,
    GlobalDomainPhase,
    decode_comm_init,
    decode_descriptor_table,
    decode_domain_command,
    encode_comm_init,
    encode_comm_init_result,
    encode_descriptor_table,
    encode_domain_command,
    resolve_global_comm_capability,
    validate_descriptor_table,
)


def _members() -> tuple[GlobalDomainMember, ...]:
    return (
        GlobalDomainMember(0, 0, 3, 0),
        GlobalDomainMember(1, 0, 7, 1),
    )


def _descriptors() -> tuple[GlobalDomainDescriptor, ...]:
    return tuple(
        GlobalDomainDescriptor(
            version=GLOBAL_DOMAIN_VERSION,
            profile_id=GLOBAL_DOMAIN_PROFILE_IDS["sim"],
            domain_rank=rank,
            rank_count=2,
            mapping_size=4096,
            handle=f"/simpler-test-{rank}".encode(),
        )
        for rank in range(2)
    )


@pytest.mark.parametrize(
    ("platform", "profile"),
    (
        ("a2a3sim", "sim"),
        ("a5sim", "sim"),
        ("a2a3", "a3-fabric-v1"),
    ),
)
def test_global_comm_capability_reports_only_implemented_backends(platform, profile):
    result = resolve_global_comm_capability(platform=platform, profile=profile, local_device_count=2)

    assert result.profile == profile
    assert result.max_ranks == 64
    assert result.descriptor_bytes == GLOBAL_DOMAIN_DESCRIPTOR_BYTES
    assert result.local_device_count == 2


@pytest.mark.parametrize(
    ("platform", "profile"),
    (
        ("a2a3", "sim"),
        ("a2a3sim", "a3-fabric-v1"),
        ("a5", "sim"),
        ("a5", "a3-fabric-v1"),
    ),
)
def test_global_comm_capability_rejects_unimplemented_backends(platform, profile):
    with pytest.raises(ValueError, match="Global CommDomain is not supported"):
        resolve_global_comm_capability(platform=platform, profile=profile, local_device_count=2)


def test_local_l3_comm_init_rejects_unsupported_capability_without_caching_topology():
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415
    from simpler.worker import Worker, _GlobalNodeRuntime, _run_local_global_domain_control  # noqa: PLC0415

    inner_worker = Worker(level=3, num_sub_workers=0)
    runtime = _GlobalNodeRuntime(
        worker_id=0,
        device_ids=(0,),
        platform="a5",
        comm_profile="sim",
        global_device_ranks=(0,),
        node_rank=0,
        node_count=1,
        cluster_id="cluster",
        is_remote=False,
    )
    comm_inits = {}
    command = GlobalCommInitCommand(
        cluster_id="cluster",
        topology_hash="topology",
        profile="sim",
        node_rank=0,
        node_count=1,
        members=(GlobalDomainMember(0, 0, 0, 0),),
    )

    try:
        with pytest.raises(ValueError, match="Global CommDomain is not supported"):
            _run_local_global_domain_control(
                inner_worker,
                runtime,
                comm_inits,
                ControlName.COMM_INIT,
                encode_comm_init(command),
            )

        assert comm_inits == {}
    finally:
        inner_worker.close()


def test_global_domain_wire_round_trips_topology_and_descriptor_table():
    init = GlobalCommInitCommand("cluster", "topology", "sim", 0, 2, _members())
    command = GlobalDomainCommand(
        phase=GlobalDomainPhase.IMPORT,
        domain_id=11,
        generation=1,
        name="tp",
        profile="sim",
        window_size=2048,
        members=_members(),
        buffers=(GlobalDomainBuffer("payload", 128),),
        descriptors=_descriptors(),
    )

    assert decode_comm_init(encode_comm_init(init)) == init
    assert decode_domain_command(encode_domain_command(command)) == command
    assert decode_descriptor_table(encode_descriptor_table(_descriptors())) == _descriptors()
    assert GLOBAL_DOMAIN_DESCRIPTOR_BYTES == 288


def _failure_injection_worker(*, platform: str = "a2a3sim", profile: str = "sim"):
    from simpler.worker import RemoteWorkerSpec, Worker, _RunResources  # noqa: PLC0415

    worker = Worker(level=4, num_sub_workers=0)
    node_ids = tuple(
        worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=f"127.0.0.1:{19073 + index}",
                platform=platform,
                device_ids=(0,),
                comm_profile=profile,
                global_device_ranks=(index,),
            )
        )
        for index in range(2)
    )
    resources = _RunResources()
    worker._worker = object()
    worker._building_run_resources = resources
    return worker, resources, node_ids


def _mpi_static_worker():
    from simpler.worker import MpiL3GroupSpec, Worker, _RunResources  # noqa: PLC0415

    worker = Worker(level=4, num_sub_workers=0)
    node_ids = worker.add_mpirun_worker_group(
        MpiL3GroupSpec(
            hosts=("127.0.0.1", "127.0.0.1"),
            platform="a2a3sim",
            command_port_base=21073,
            health_port_base=22073,
            device_ids_by_rank=((0,), (0,)),
            comm_profile="sim",
            global_device_ranks_by_rank=((0,), (1,)),
        )
    )
    resources = _RunResources()
    worker._worker = object()
    worker._building_run_resources = resources
    return worker, resources, node_ids


def _install_global_domain_failure_injector(monkeypatch, worker, *, fail_phase, fail_node):
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415

    calls = []

    def control(worker_id, control_name, payload):
        control_name = ControlName(control_name)
        if control_name is ControlName.COMM_INIT:
            init = decode_comm_init(payload)
            calls.append(("COMM_INIT", worker_id))
            return encode_comm_init_result(
                resolve_global_comm_capability(
                    platform="a2a3sim",
                    profile=init.profile,
                    local_device_count=1,
                )
            )

        assert control_name is ControlName.ALLOC_DOMAIN
        command = decode_domain_command(payload)
        calls.append((command.phase, worker_id))
        if command.phase is fail_phase and worker_id == fail_node:
            raise RuntimeError(f"injected {command.phase.name} failure")
        if command.phase is not GlobalDomainPhase.PREPARE_EXPORT:
            return b""
        descriptors = tuple(
            GlobalDomainDescriptor(
                version=GLOBAL_DOMAIN_VERSION,
                profile_id=GLOBAL_DOMAIN_PROFILE_IDS[command.profile],
                domain_rank=member.domain_rank,
                rank_count=len(command.members),
                mapping_size=4096,
                handle=f"/injected-{member.domain_rank}".encode(),
            )
            for member in command.members
            if member.node_worker_id == worker_id
        )
        return encode_descriptor_table(descriptors)

    monkeypatch.setattr(worker, "_global_domain_control", control)
    return calls


def test_mpirun_group_global_domain_uses_mpi_prepare_commit_without_l4_import(monkeypatch):
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    worker, resources, node_ids = _mpi_static_worker()
    calls = []

    def control(worker_id, control_name, payload):
        control_name = ControlName(control_name)
        if control_name is ControlName.COMM_INIT:
            init = decode_comm_init(payload)
            calls.append(("COMM_INIT", worker_id))
            return encode_comm_init_result(
                resolve_global_comm_capability(
                    platform="a2a3sim",
                    profile=init.profile,
                    local_device_count=1,
                )
            )
        assert control_name is ControlName.ALLOC_DOMAIN
        command = decode_domain_command(payload)
        calls.append((command.phase, worker_id))
        if command.phase is GlobalDomainPhase.PREPARE_EXPORT:
            descriptors = tuple(
                GlobalDomainDescriptor(
                    version=GLOBAL_DOMAIN_VERSION,
                    profile_id=GLOBAL_DOMAIN_PROFILE_IDS[command.profile],
                    domain_rank=member.domain_rank,
                    rank_count=len(command.members),
                    mapping_size=4096,
                    handle=f"/mpi-prepared-{member.domain_rank}".encode(),
                )
                for member in command.members
            )
            return encode_descriptor_table(descriptors)
        if command.phase is GlobalDomainPhase.IMPORT:
            raise RuntimeError("L4 broker IMPORT should not run for a full mpirun group")
        return b""

    monkeypatch.setattr(worker, "_global_domain_control", control)
    try:
        handle = worker._allocate_global_domain(
            name="mpi-static",
            members=((node_ids[0], 0), (node_ids[1], 0)),
            window_size=4096,
            buffers=[CommBufferSpec("payload", "uint8", 4096, 4096)],
            retain_after_run=False,
        )

        assert handle.mapping_size == 4096
        assert handle.members[0].global_device_rank == 0
        assert handle.members[1].global_device_rank == 1
        counts = Counter(phase for phase, _worker_id in calls)
        assert counts["COMM_INIT"] == 2
        assert counts[GlobalDomainPhase.PREPARE_EXPORT] == 2
        assert counts[GlobalDomainPhase.COMMIT] == 2
        assert counts[GlobalDomainPhase.IMPORT] == 0
        assert worker._live_global_domains["mpi-static"] is handle
        assert resources.live_global_domains["mpi-static"] is handle
    finally:
        _close_failure_injection_worker(worker, resources)


def _close_failure_injection_worker(worker, resources):
    worker._building_run_resources = None
    worker._live_global_domains.clear()
    resources.live_global_domains.clear()
    worker._worker = None
    worker.close()


@pytest.mark.parametrize(
    "fail_phase",
    (
        GlobalDomainPhase.PREPARE_EXPORT,
        GlobalDomainPhase.IMPORT,
        GlobalDomainPhase.COMMIT,
    ),
)
def test_global_domain_transaction_aborts_all_prepared_nodes_after_phase_failure(monkeypatch, fail_phase):
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    worker, resources, node_ids = _failure_injection_worker()
    calls = _install_global_domain_failure_injector(
        monkeypatch,
        worker,
        fail_phase=fail_phase,
        fail_node=node_ids[1],
    )
    try:
        with pytest.raises(RuntimeError, match=f"injected {fail_phase.name} failure"):
            worker._allocate_global_domain(
                name="failure-injection",
                members=((node_ids[0], 0), (node_ids[1], 0)),
                window_size=4096,
                buffers=[CommBufferSpec("payload", "uint8", 4096, 4096)],
                retain_after_run=False,
            )

        abort_nodes = [node_id for phase, node_id in calls if phase is GlobalDomainPhase.ABORT]
        assert abort_nodes == list(node_ids)
        assert worker._live_global_domains == {}
        assert resources.live_global_domains == {}
    finally:
        _close_failure_injection_worker(worker, resources)


def test_global_domain_abort_failure_preserves_primary_error_and_continues_cleanup(monkeypatch):
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    worker, resources, node_ids = _failure_injection_worker()
    calls = _install_global_domain_failure_injector(
        monkeypatch,
        worker,
        fail_phase=GlobalDomainPhase.IMPORT,
        fail_node=node_ids[1],
    )
    original_control = worker._global_domain_control

    def fail_first_abort(worker_id, control_name, payload):
        if ControlName(control_name) is ControlName.ALLOC_DOMAIN:
            command = decode_domain_command(payload)
            if command.phase is GlobalDomainPhase.ABORT and worker_id == node_ids[0]:
                calls.append((command.phase, worker_id))
                raise RuntimeError("injected ABORT failure")
        return original_control(worker_id, control_name, payload)

    monkeypatch.setattr(worker, "_global_domain_control", fail_first_abort)
    try:
        with pytest.raises(RuntimeError, match="injected IMPORT failure"):
            worker._allocate_global_domain(
                name="abort-failure-injection",
                members=((node_ids[0], 0), (node_ids[1], 0)),
                window_size=4096,
                buffers=[CommBufferSpec("payload", "uint8", 4096, 4096)],
                retain_after_run=False,
            )

        abort_nodes = [node_id for phase, node_id in calls if phase is GlobalDomainPhase.ABORT]
        assert abort_nodes == list(node_ids)
        assert worker._live_global_domains == {}
        assert resources.live_global_domains == {}
    finally:
        _close_failure_injection_worker(worker, resources)


def test_allocate_global_domain_rejects_unsupported_capability_before_control(monkeypatch):
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    worker, resources, node_ids = _failure_injection_worker(platform="a5", profile="sim")
    calls = []
    monkeypatch.setattr(worker, "_global_domain_control", lambda *args: calls.append(args))
    try:
        with pytest.raises(ValueError, match="Global CommDomain is not supported"):
            worker._allocate_global_domain(
                name="unsupported",
                members=((node_ids[0], 0), (node_ids[1], 0)),
                window_size=4096,
                buffers=[CommBufferSpec("payload", "uint8", 4096, 4096)],
                retain_after_run=False,
            )

        assert calls == []
        assert worker._live_global_domains == {}
        assert resources.live_global_domains == {}
    finally:
        _close_failure_injection_worker(worker, resources)


def test_global_domain_descriptor_table_rejects_different_mapping_sizes():
    descriptors = list(_descriptors())
    descriptors[1] = GlobalDomainDescriptor(
        version=GLOBAL_DOMAIN_VERSION,
        profile_id=GLOBAL_DOMAIN_PROFILE_IDS["sim"],
        domain_rank=1,
        rank_count=2,
        mapping_size=8192,
        handle=b"/simpler-test-1",
    )

    with pytest.raises(ValueError, match="mapping sizes differ"):
        validate_descriptor_table(tuple(descriptors), rank_count=2, profile="sim")


def test_global_domain_release_retries_after_callback_failure():
    from simpler.task_interface import GlobalCommDomainHandle  # noqa: PLC0415

    attempts = 0

    def release_fn(_handle):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient release failure")

    handle = GlobalCommDomainHandle(
        name="retry",
        members=(),
        buffers=(),
        domain_id=17,
        generation=1,
        mapping_size=4096,
        retain_after_run=False,
        _release_fn=release_fn,
    )

    with pytest.raises(RuntimeError, match="transient release failure"):
        handle.release()

    assert not handle.released
    handle.release()
    assert handle.released
    assert attempts == 2


def test_old_global_domain_release_does_not_remove_same_name_replacement():
    from simpler.task_interface import GlobalCommDomainHandle  # noqa: PLC0415
    from simpler.worker import Worker, _RunResources  # noqa: PLC0415

    worker = Worker(level=4, num_sub_workers=0)
    resources = _RunResources()

    def make_handle(domain_id: int) -> GlobalCommDomainHandle:
        return GlobalCommDomainHandle(
            name="reuse",
            members=(),
            buffers=(),
            domain_id=domain_id,
            generation=1,
            mapping_size=4096,
            retain_after_run=False,
            _release_fn=worker._release_global_domain_handle,
        )

    first = make_handle(17)
    second = make_handle(18)
    worker._worker = object()
    worker._building_run_resources = resources
    worker._live_global_domains[first.name] = first
    resources.live_global_domains[first.name] = first
    try:
        first.release()
        worker._live_global_domains[second.name] = second
        resources.live_global_domains[second.name] = second

        worker._execute_pending_global_domain_releases(resources)

        assert first.freed
        assert worker._live_global_domains[second.name] is second
        assert resources.live_global_domains[second.name] is second
    finally:
        worker._building_run_resources = None
        worker._live_global_domains.clear()
        resources.live_global_domains.clear()
        worker._worker = None
        worker.close()


def test_global_domain_backend_release_failure_is_terminal(monkeypatch):
    from simpler.task_interface import GlobalCommDomainHandle  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    attempts = 0
    worker = Worker(level=4, num_sub_workers=0)
    worker._worker = object()

    def fail_control(_worker_id, _control_name, _payload):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("partial backend release")

    monkeypatch.setattr(worker, "_global_domain_control", fail_control)
    handle = GlobalCommDomainHandle(
        name="terminal",
        members=(_members()[0],),
        buffers=(),
        domain_id=19,
        generation=1,
        mapping_size=4096,
        retain_after_run=False,
        _release_fn=worker._release_global_domain_handle,
    )
    try:
        with pytest.raises(RuntimeError, match="partial backend release"):
            worker._free_global_domain_after_fence(handle)
        with pytest.raises(RuntimeError, match="partial backend release"):
            worker._free_global_domain_after_fence(handle)

        assert attempts == 1
        assert not handle.freed
        assert worker._failed_global_domain_releases[handle.domain_id] is handle
    finally:
        worker._failed_global_domain_releases.clear()
        worker._worker = None
        worker.close()


def test_childless_host_buffer_uses_dedicated_exception():
    from simpler.worker import Worker, _NoHostBufferChildrenError  # noqa: PLC0415

    worker = Worker(level=3, num_sub_workers=0)
    try:
        with pytest.raises(_NoHostBufferChildrenError, match="at least one forked chip or sub child"):
            worker._create_host_buffer_locked(64)
    finally:
        worker.close()


def test_remote_compute_orch_submits_local_l2_add(monkeypatch):
    from simpler.global_comm_smoke import remote_compute_orch  # noqa: PLC0415
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415
    from simpler.task_interface import CallConfig, TaskArgs, TensorArgType  # noqa: PLC0415

    digest = bytes(range(32))
    chip_handle = object()
    monkeypatch.setattr(
        "simpler.remote_l3_session.get_inner_handle",
        lambda hashid: chip_handle if hashid == digest.hex() else get_inner_handle(hashid),
    )

    class FakeContext:
        buffer_ptrs = {"lhs": 0x1000, "rhs": 0x2000, "input": 0x3000}

    class FakeOrchestrator:
        def __init__(self):
            self.submitted = None

        def get_global_domain(self, domain_id):
            assert domain_id == 17
            return {0: FakeContext()}

        def submit_next_level(self, handle, task_args, cfg, *, worker):
            self.submitted = (handle, task_args, cfg, worker)

    args = TaskArgs()
    args.add_scalar(17)
    args.add_scalar(0)
    for offset in range(0, 32, 8):
        args.add_scalar(int.from_bytes(digest[offset : offset + 8], "little"))
    config = CallConfig()
    orch = FakeOrchestrator()

    remote_compute_orch(orch, args, config)  # type: ignore[arg-type]

    assert orch.submitted is not None
    handle, chip_args, submitted_config, worker_id = orch.submitted
    assert handle is chip_handle
    assert submitted_config is config
    assert worker_id == 0
    assert chip_args.tensor_count() == 3
    assert [chip_args.tensor(index).data for index in range(3)] == [0x1000, 0x2000, 0x3000]
    assert [chip_args.tensor(index).child_memory for index in range(3)] == [True, True, True]
    assert [chip_args.tag(index) for index in range(3)] == [
        TensorArgType.INPUT,
        TensorArgType.INPUT,
        TensorArgType.OUTPUT_EXISTING,
    ]


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_tcp_ports(ports: tuple[int, ...], timeout_s: float = 5.0) -> None:
    pending = set(ports)
    deadline = time.monotonic() + timeout_s
    while pending:
        for port in tuple(pending):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"remote L3 daemons did not become ready on ports {sorted(pending)}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=min(0.1, remaining)):
                    pending.remove(port)
            except OSError:
                pass
        if pending:
            time.sleep(0.01)


@pytest.mark.skipif(os.name == "nt", reason="hierarchical workers require fork")
def test_two_remote_daemons_build_and_copy_global_domain_without_mpirun():
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415
    from simpler.worker import RemoteWorkerSpec, Worker  # noqa: PLC0415

    ports = (_free_tcp_port(), _free_tcp_port())
    daemons = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "simpler.remote_l3_worker",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for port in ports
    ]
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=20)
    captured: dict[str, object] = {}
    try:
        _wait_for_tcp_ports(ports)
        node_ids = tuple(
            worker.add_remote_worker(
                RemoteWorkerSpec(
                    endpoint=f"127.0.0.1:{port}",
                    platform="a2a3sim",
                    device_ids=(0,),
                    comm_profile="sim",
                )
            )
            for port in ports
        )
        worker.init()

        def parent_orch(orch, _args, _cfg):
            domain = orch.allocate_global_domain(
                name="tcp-global",
                members=((node_ids[0], 0), (node_ids[1], 0)),
                window_size=4096,
                buffers=(CommBufferSpec("payload", "uint8", 64, 64),),
                retain_after_run=True,
            )
            orch.copy_to_global_domain(domain, 0, b"node-zero", buffer="payload")
            orch.copy_to_global_domain(domain, 1, b"node-one", buffer="payload")
            captured["ranks"] = tuple(member.global_device_rank for member in domain.members)
            captured["handle"] = domain

        worker.run(parent_orch)
        assert not captured["handle"].freed

        def read_orch(orch, _args, _cfg):
            domain = captured["handle"]
            try:
                captured["rank0"] = orch.copy_from_global_domain(domain, 0, len(b"node-zero"), buffer="payload")
                captured["rank1"] = orch.copy_from_global_domain(domain, 1, len(b"node-one"), buffer="payload")
            finally:
                domain.release()

        worker.run(read_orch)
        assert captured["rank0"] == b"node-zero"
        assert captured["rank1"] == b"node-one"
        assert captured["ranks"] == (0, 1)
        assert captured["handle"].freed
    finally:
        worker.close()
        for daemon in daemons:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="hierarchical workers require fork")
def test_local_and_remote_l3_build_and_copy_global_domain_without_mpirun():
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415
    from simpler.worker import RemoteWorkerSpec, Worker  # noqa: PLC0415

    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "simpler.remote_l3_worker",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=20)
    captured: dict[str, object] = {}
    try:
        _wait_for_tcp_ports((port,))
        local_node_id = worker.add_worker(
            Worker(
                level=3,
                device_ids=[0],
                num_sub_workers=0,
                platform="a2a3sim",
                runtime="tensormap_and_ringbuffer",
                comm_profile="sim",
                global_device_ranks=(0,),
            )
        )
        remote_node_id = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=f"127.0.0.1:{port}",
                platform="a2a3sim",
                device_ids=(0,),
                comm_profile="sim",
                global_device_ranks=(1,),
            )
        )
        worker.init()

        def build_orch(orch, _args, _cfg):
            domain = orch.allocate_global_domain(
                name="mixed-global",
                members=((local_node_id, 0), (remote_node_id, 0)),
                window_size=4096,
                buffers=(CommBufferSpec("payload", "uint8", 64, 64),),
                retain_after_run=True,
            )
            orch.copy_to_global_domain(domain, 0, b"local-l3", buffer="payload")
            orch.copy_to_global_domain(domain, 1, b"remote-l3", buffer="payload")
            captured["ranks"] = tuple(member.global_device_rank for member in domain.members)
            captured["domain"] = domain

        worker.run(build_orch)
        domain = captured["domain"]
        assert not domain.freed

        def read_orch(orch, _args, _cfg):
            try:
                captured["local"] = orch.copy_from_global_domain(
                    domain,
                    0,
                    len(b"local-l3"),
                    buffer="payload",
                )
                captured["remote"] = orch.copy_from_global_domain(
                    domain,
                    1,
                    len(b"remote-l3"),
                    buffer="payload",
                )
            finally:
                domain.release()

        worker.run(read_orch)
        assert captured["local"] == b"local-l3"
        assert captured["remote"] == b"remote-l3"
        assert captured["ranks"] == (0, 1)
        assert domain.freed
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)
