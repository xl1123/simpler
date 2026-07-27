# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Launch and compare the real socket and MPI-sidecar L4 paths."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("topology config must be a JSON object")
    return config


def _python_env(repo_root: str) -> dict[str, str]:
    env = dict(os.environ)
    entries = [repo_root, str(Path(repo_root) / "python")]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _wait_path(path: Path, processes: list[subprocess.Popen[Any]], timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"{label}: process {process.args!r} exited with {process.returncode}")
        time.sleep(0.05)
    raise TimeoutError(f"{label}: {path} was not created")


def _wait_log(path: Path, text: str, process: subprocess.Popen[Any], timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists() and text in path.read_text(encoding="utf-8", errors="replace"):
            return
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited with {process.returncode}; see {path}")
        time.sleep(0.1)
    raise TimeoutError(f"{label} did not log {text!r}; see {path}")


def _wait_tcp(host: str, port: int, process: subprocess.Popen[Any], timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited with {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=min(0.2, timeout_s)):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"{label} did not listen on {host}:{port}")


def _wait_tcp_endpoint(host: str, port: int, timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=min(0.5, timeout_s)):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"{label} did not listen on {host}:{port}")


def _terminate(process: subprocess.Popen[Any], timeout_s: float = 5.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_s)


def _run_case(
    *,
    python: str,
    env: dict[str, str],
    transport: str,
    daemon_endpoint: str,
    bootstrap_path: Path,
    platform: str,
    timeout_s: float,
) -> dict[str, Any]:
    command = [
        python,
        str(REPO_ROOT / "tools/mpi_l4_sidecar/e2e_case.py"),
        "--control-transport",
        transport,
        "--daemon-endpoint",
        daemon_endpoint,
        "--platform",
        platform,
        "--timeout",
        str(timeout_s),
    ]
    if transport == "mpi_sidecar":
        command += ["--sidecar-endpoint", str(bootstrap_path), "--mpi-rank", "1"]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout_s + 15.0)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"{transport} validation failed with {result.returncode}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"{transport} validation emitted no JSON result")
    return json.loads(lines[-1])


def _run_npu_case(
    *,
    python: str,
    env: dict[str, str],
    transport: str,
    remotes: list[dict[str, Any]],
    bootstrap_path: Path,
    platform: str,
    runtime: str,
    timeout_s: float,
) -> dict[str, Any]:
    command = [
        python,
        str(REPO_ROOT / "tools/mpi_l4_sidecar/npu_e2e_case.py"),
        "--control-transport",
        transport,
        "--machine-a",
        str(remotes[0]["daemon_endpoint"]),
        "--machine-b",
        str(remotes[1]["daemon_endpoint"]),
        "--machine-a-devices",
        ",".join(str(item) for item in remotes[0]["device_ids"]),
        "--machine-b-devices",
        ",".join(str(item) for item in remotes[1]["device_ids"]),
        "--machine-a-mpi-rank",
        str(remotes[0]["rank"]),
        "--machine-b-mpi-rank",
        str(remotes[1]["rank"]),
        "--platform",
        platform,
        "--runtime",
        runtime,
        "--timeout",
        str(timeout_s),
    ]
    if transport == "mpi_sidecar":
        command += ["--sidecar-endpoint", str(bootstrap_path)]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout_s + 180.0)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"{transport} NPU validation failed with {result.returncode}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"{transport} NPU validation emitted no JSON result")
    return json.loads(lines[-1])


def _compare(legacy: dict[str, Any], mpi: dict[str, Any]) -> None:
    fields = tuple(sorted((set(legacy) | set(mpi)) - {"control_transport"}))
    differences = {
        field: (legacy.get(field), mpi.get(field))
        for field in fields
        if legacy.get(field) != mpi.get(field)
    }
    if differences:
        raise RuntimeError(f"legacy/MPI L4 result mismatch: {differences}")
    print(json.dumps({"event": "legacy_vs_mpi", "status": "PASS", "compared": fields}, sort_keys=True))


def _frame_records(path: Path, event: str) -> Counter[tuple[Any, ...]]:
    records: Counter[tuple[Any, ...]] = Counter()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != event or record.get("lane") != "COMMAND":
            continue
        key = (
            int(record["session_id"]),
            int(record["worker_id"]),
            int(record["frame_type"]),
            int(record["sequence"]),
            str(record["sha256"]),
        )
        records[key] += 1
    return records


def _verify_frame_logs(rank0_log: Path, target_logs: list[Path]) -> None:
    l4_sent = _frame_records(rank0_log, "L4_TO_MPI")
    l3_received: Counter[tuple[Any, ...]] = Counter()
    l3_sent: Counter[tuple[Any, ...]] = Counter()
    for path in target_logs:
        l3_received.update(_frame_records(path, "MPI_TO_L3"))
        l3_sent.update(_frame_records(path, "L3_TO_MPI"))
    l4_received = _frame_records(rank0_log, "MPI_TO_L4")
    if l4_sent != l3_received:
        raise RuntimeError("L4->L3 MPI command frame sequence/hash comparison failed")
    if l3_sent != l4_received:
        raise RuntimeError("L3->L4 MPI command frame sequence/hash comparison failed")
    if not any(key[2] == 2 for key in l4_sent) or not any(key[2] == 5 for key in l3_sent):
        raise RuntimeError("MPI command log is missing TASK or COMPLETION")
    print(
        json.dumps(
            {
                "event": "mpi_frame_sequence_hash_compare",
                "status": "PASS",
                "l4_to_l3_frames": sum(l4_sent.values()),
                "l3_to_l4_frames": sum(l3_sent.values()),
            },
            sort_keys=True,
        )
    )


def _stop_world(python: str, env: dict[str, str], bootstrap_path: Path, timeout_s: float) -> None:
    subprocess.run(
        [python, "-m", "simpler.remote_l3_sidecar_proxy", "--stop-world", str(bootstrap_path)],
        env=env,
        check=True,
        timeout=timeout_s,
    )


def _start_local(
    command: list[str], *, env: dict[str, str], log_path: Path, processes: list[subprocess.Popen[Any]]
) -> subprocess.Popen[Any]:
    log = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    processes.append(process)
    return process


def _local_proxy_command(
    python: str, rank: int, sidecar_path: Path, session_dir: Path, worker_ids: list[int], bootstrap_path: Path
) -> list[str]:
    command = [
        python,
        "-m",
        "simpler.remote_l3_sidecar_proxy",
        "--rank",
        str(rank),
        "--sidecar-socket",
        str(sidecar_path),
        "--session-dir",
        str(session_dir),
        "--worker-ids",
        ",".join(str(item) for item in worker_ids),
    ]
    if rank == 0:
        command += ["--bootstrap-socket", str(bootstrap_path)]
    return command


def _validate_common(config: dict[str, Any]) -> tuple[str, list[str], str, float]:
    python = str(config.get("python", sys.executable))
    mpi_command = config.get("mpi_command")
    if not isinstance(mpi_command, list) or not mpi_command:
        raise ValueError("config.mpi_command must be the full mpirun prefix as a JSON string array")
    sidecar_binary = str(config["sidecar_binary"])
    timeout_s = float(config.get("timeout_s", 30.0))
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    return python, [str(item) for item in mpi_command], sidecar_binary, timeout_s


def run_one_host(config: dict[str, Any]) -> int:
    python, mpi_command, sidecar_binary, timeout_s = _validate_common(config)
    daemon_endpoint = str(config.get("daemon_endpoint", "127.0.0.1:19073"))
    daemon_host, separator, daemon_port = daemon_endpoint.rpartition(":")
    if not separator or not daemon_host:
        raise ValueError("daemon_endpoint must be host:port")
    platform = str(config.get("platform", "a2a3sim"))
    base_dir = Path(config.get("work_dir", tempfile.gettempdir()))
    job_dir = base_dir / f"simpler-mpi-l4-{uuid.uuid4().hex[:12]}"
    job_dir.mkdir(mode=0o700, parents=True)
    bootstrap_path = job_dir / "l4-bootstrap.sock"
    proxy_template = str(job_dir / "proxy.%r.sock")
    env = _python_env(str(REPO_ROOT))
    processes: list[subprocess.Popen[Any]] = []
    proxies: list[subprocess.Popen[Any]] = []
    sidecar: subprocess.Popen[Any] | None = None
    try:
        daemon = _start_local(
            [python, "-m", "simpler.remote_l3_worker", "--host", daemon_host, "--port", daemon_port],
            env=env,
            log_path=job_dir / "daemon.log",
            processes=processes,
        )
        _wait_tcp(daemon_host, int(daemon_port), daemon, timeout_s, "remote L3 daemon")
        for rank, worker_ids in ((0, []), (1, [0])):
            proxy_path = job_dir / f"proxy.{rank}.sock"
            proxy = _start_local(
                _local_proxy_command(
                    python, rank, proxy_path, job_dir / f"sessions.{rank}", worker_ids, bootstrap_path
                ),
                env=env,
                log_path=job_dir / f"proxy.{rank}.log",
                processes=processes,
            )
            proxies.append(proxy)
            _wait_path(proxy_path, [proxy], timeout_s, f"rank {rank} proxy")
        sidecar_command = mpi_command + [
            sidecar_binary,
            "--proxy-socket-template",
            proxy_template,
            "--topology-id",
            "one-host-two-rank",
            "--worker-map",
            "0:;1:0",
        ]
        sidecar = _start_local(sidecar_command, env=env, log_path=job_dir / "sidecar.log", processes=processes)
        _wait_path(bootstrap_path, [sidecar], timeout_s, "MPI world/L4 bootstrap")

        legacy = _run_case(
            python=python,
            env=env,
            transport="socket",
            daemon_endpoint=daemon_endpoint,
            bootstrap_path=bootstrap_path,
            platform=platform,
            timeout_s=timeout_s,
        )
        mpi = _run_case(
            python=python,
            env=env,
            transport="mpi_sidecar",
            daemon_endpoint=daemon_endpoint,
            bootstrap_path=bootstrap_path,
            platform=platform,
            timeout_s=timeout_s,
        )
        _compare(legacy, mpi)
        _verify_frame_logs(job_dir / "proxy.0.log", [job_dir / "proxy.1.log"])
        _stop_world(python, env, bootstrap_path, timeout_s)
        sidecar.wait(timeout=timeout_s)
        if sidecar.returncode != 0:
            raise RuntimeError(f"MPI sidecar exited with {sidecar.returncode}; see {job_dir / 'sidecar.log'}")
        for proxy in proxies:
            proxy.wait(timeout=timeout_s)
            if proxy.returncode != 0:
                raise RuntimeError(f"MPI proxy exited with {proxy.returncode}; see {job_dir}")
        residue = list(job_dir.rglob("*.sock"))
        if residue:
            raise RuntimeError(f"Unix socket residue remains: {residue}")
        print(f"MPI sidecar phase-1 validation PASS; logs: {job_dir}")
        return 0
    finally:
        for process in reversed(processes):
            _terminate(process)


def _remote_command(host: str, command: list[str], env: dict[str, str] | None = None) -> list[str]:
    assignments = [] if env is None else [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    remote = "exec env " + " ".join(assignments + [shlex.join(command)])
    return ["ssh", host, remote]


def run_two_host(config: dict[str, Any]) -> int:
    python, mpi_command, sidecar_binary, timeout_s = _validate_common(config)
    hosts = config.get("hosts")
    if not isinstance(hosts, list) or len(hosts) != 2:
        raise ValueError("two-host config requires exactly two hosts")
    by_rank = {int(host["rank"]): host for host in hosts}
    if set(by_rank) != {0, 1} or not bool(by_rank[0].get("local", False)):
        raise ValueError("two-host phase-1 requires rank 0 local and ranks exactly 0,1")
    common_work_dir = str(config.get("work_dir", "/tmp"))
    job_dir = Path(common_work_dir) / f"simpler-mpi-l4-{uuid.uuid4().hex[:12]}"
    job_dir.mkdir(mode=0o700, parents=True)
    bootstrap_path = job_dir / "l4-bootstrap.sock"
    proxy_template = str(job_dir / "proxy.%r.sock")
    platform = str(config.get("platform", "a2a3sim"))
    daemon_endpoint = str(by_rank[1]["daemon_endpoint"])
    env = _python_env(str(REPO_ROOT))
    processes: list[subprocess.Popen[Any]] = []
    try:
        rank0_proxy = _start_local(
            _local_proxy_command(
                python,
                0,
                Path(proxy_template.replace("%r", "0")),
                job_dir / "sessions.0",
                [],
                bootstrap_path,
            ),
            env=env,
            log_path=job_dir / "proxy.0.log",
            processes=processes,
        )
        _wait_path(Path(proxy_template.replace("%r", "0")), [rank0_proxy], timeout_s, "rank 0 proxy")

        remote = by_rank[1]
        remote_host = str(remote["ssh"])
        remote_python = str(remote.get("python", python))
        remote_repo = str(remote["repo_root"])
        remote_env = _python_env(remote_repo)
        remote_proxy_path = proxy_template.replace("%r", "1")
        remote_proxy_log = job_dir / "proxy.1.ssh.log"
        proxy_command = _local_proxy_command(
            remote_python, 1, Path(remote_proxy_path), job_dir / "sessions.1", [0], bootstrap_path
        )
        remote_proxy = _start_local(
            _remote_command(remote_host, proxy_command, {"PYTHONPATH": remote_env["PYTHONPATH"]}),
            env=env,
            log_path=remote_proxy_log,
            processes=processes,
        )
        _wait_log(remote_proxy_log, "LISTENING", remote_proxy, timeout_s, "rank 1 proxy")
        daemon_host, separator, daemon_port = daemon_endpoint.rpartition(":")
        if not separator:
            raise ValueError("rank 1 daemon_endpoint must be host:port")
        remote_daemon = _start_local(
            _remote_command(
                remote_host,
                [
                    remote_python,
                    "-m",
                    "simpler.remote_l3_worker",
                    "--host",
                    daemon_host,
                    "--port",
                    daemon_port,
                ],
                {"PYTHONPATH": remote_env["PYTHONPATH"]},
            ),
            env=env,
            log_path=job_dir / "daemon.1.ssh.log",
            processes=processes,
        )
        _wait_tcp(daemon_host, int(daemon_port), remote_daemon, timeout_s, "rank 1 remote L3 daemon")
        sidecar = _start_local(
            mpi_command
            + [
                sidecar_binary,
                "--proxy-socket-template",
                proxy_template,
                "--topology-id",
                str(config.get("topology_id", "two-host-two-rank")),
                "--worker-map",
                "0:;1:0",
            ],
            env=env,
            log_path=job_dir / "sidecar.log",
            processes=processes,
        )
        _wait_path(bootstrap_path, [sidecar], timeout_s, "MPI world/L4 bootstrap")
        legacy = _run_case(
            python=python,
            env=env,
            transport="socket",
            daemon_endpoint=daemon_endpoint,
            bootstrap_path=bootstrap_path,
            platform=platform,
            timeout_s=timeout_s,
        )
        mpi = _run_case(
            python=python,
            env=env,
            transport="mpi_sidecar",
            daemon_endpoint=daemon_endpoint,
            bootstrap_path=bootstrap_path,
            platform=platform,
            timeout_s=timeout_s,
        )
        _compare(legacy, mpi)
        _verify_frame_logs(job_dir / "proxy.0.log", [remote_proxy_log])
        _stop_world(python, env, bootstrap_path, timeout_s)
        sidecar.wait(timeout=timeout_s)
        if sidecar.returncode != 0:
            raise RuntimeError(f"MPI sidecar exited with {sidecar.returncode}; see {job_dir / 'sidecar.log'}")
        rank0_proxy.wait(timeout=timeout_s)
        remote_proxy.wait(timeout=timeout_s)
        if rank0_proxy.returncode != 0 or remote_proxy.returncode != 0:
            raise RuntimeError(f"MPI proxy shutdown failed; see {job_dir}")
        residue = list(job_dir.rglob("*.sock"))
        if residue:
            raise RuntimeError(f"Unix socket residue remains: {residue}")
        remote_residue = subprocess.run(
            [
                "ssh",
                remote_host,
                shlex.join(["find", str(job_dir), "-type", "s", "-print", "-quit"]),
            ],
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        if remote_residue.returncode != 0 or remote_residue.stdout.strip():
            raise RuntimeError(
                f"rank 1 residue verification failed: {remote_residue.stdout}{remote_residue.stderr}"
            )
        print(f"two-host MPI sidecar phase-1 validation PASS; logs: {job_dir}")
        return 0
    finally:
        for process in reversed(processes):
            _terminate(process)


def run_npu(config: dict[str, Any]) -> int:
    python, mpi_command, sidecar_binary, timeout_s = _validate_common(config)
    hosts = config.get("hosts")
    if not isinstance(hosts, list) or len(hosts) != 2:
        raise ValueError("NPU config requires exactly two machine/rank entries")
    by_rank = {int(host["rank"]): host for host in hosts}
    if set(by_rank) != {0, 1} or not bool(by_rank[0].get("local", False)):
        raise ValueError("NPU phase-1 requires local/master rank 0 and remote rank 1")
    remotes = [by_rank[0], by_rank[1]]
    for remote in remotes:
        device_ids = remote.get("device_ids")
        if not isinstance(device_ids, list) or len(device_ids) != 2:
            raise ValueError(f"rank {remote['rank']} requires exactly two device_ids")
        endpoint = str(remote.get("daemon_endpoint", ""))
        daemon_host, separator, daemon_port = endpoint.rpartition(":")
        if not separator or not daemon_host:
            raise ValueError(f"rank {remote['rank']} daemon_endpoint must be host:port")
        _wait_tcp_endpoint(
            daemon_host,
            int(daemon_port),
            timeout_s,
            f"rank {remote['rank']} pre-started remote L3 daemon",
        )

    work_dir = str(config.get("work_dir", "/tmp"))
    job_dir = Path(work_dir) / f"simpler-mpi-l4-npu-{uuid.uuid4().hex[:12]}"
    job_dir.mkdir(mode=0o700, parents=True)
    bootstrap_path = job_dir / "l4-bootstrap.sock"
    proxy_template = str(job_dir / "proxy.%r.sock")
    platform = str(config.get("platform", "a2a3"))
    runtime = str(config.get("runtime", "tensormap_and_ringbuffer"))
    env = _python_env(str(REPO_ROOT))
    processes: list[subprocess.Popen[Any]] = []
    remote_proxy_logs: list[Path] = []
    remote_proxies: list[subprocess.Popen[Any]] = []
    try:
        rank0_proxy = _start_local(
            _local_proxy_command(
                python,
                0,
                Path(proxy_template.replace("%r", "0")),
                job_dir / "sessions.0",
                [0],
                bootstrap_path,
            ),
            env=env,
            log_path=job_dir / "proxy.0.log",
            processes=processes,
        )
        _wait_path(Path(proxy_template.replace("%r", "0")), [rank0_proxy], timeout_s, "rank 0 proxy")

        for remote in remotes[1:]:
            rank = int(remote["rank"])
            remote_host = str(remote["ssh"])
            remote_python = str(remote.get("python", python))
            remote_repo = str(remote["repo_root"])
            remote_env = _python_env(remote_repo)
            proxy_log = job_dir / f"proxy.{rank}.ssh.log"
            proxy_command = _local_proxy_command(
                remote_python,
                rank,
                Path(proxy_template.replace("%r", str(rank))),
                job_dir / f"sessions.{rank}",
                [1],
                bootstrap_path,
            )
            proxy = _start_local(
                _remote_command(remote_host, proxy_command, {"PYTHONPATH": remote_env["PYTHONPATH"]}),
                env=env,
                log_path=proxy_log,
                processes=processes,
            )
            remote_proxies.append(proxy)
            remote_proxy_logs.append(proxy_log)
            _wait_log(proxy_log, "LISTENING", proxy, timeout_s, f"rank {rank} proxy")

        worker_map = "0:0;1:1"
        sidecar = _start_local(
            mpi_command
            + [
                sidecar_binary,
                "--proxy-socket-template",
                proxy_template,
                "--topology-id",
                str(config.get("topology_id", "two-machine-master-real-npu-phase1")),
                "--worker-map",
                worker_map,
            ],
            env=env,
            log_path=job_dir / "sidecar.log",
            processes=processes,
        )
        _wait_path(bootstrap_path, [sidecar], timeout_s, "MPI world/L4 bootstrap")

        legacy = _run_npu_case(
            python=python,
            env=env,
            transport="socket",
            remotes=remotes,
            bootstrap_path=bootstrap_path,
            platform=platform,
            runtime=runtime,
            timeout_s=timeout_s,
        )
        mpi = _run_npu_case(
            python=python,
            env=env,
            transport="mpi_sidecar",
            remotes=remotes,
            bootstrap_path=bootstrap_path,
            platform=platform,
            runtime=runtime,
            timeout_s=timeout_s,
        )
        _compare(legacy, mpi)
        _verify_frame_logs(job_dir / "proxy.0.log", [job_dir / "proxy.0.log", *remote_proxy_logs])
        _stop_world(python, env, bootstrap_path, timeout_s)
        sidecar.wait(timeout=timeout_s)
        rank0_proxy.wait(timeout=timeout_s)
        for proxy in remote_proxies:
            proxy.wait(timeout=timeout_s)
        if sidecar.returncode != 0 or rank0_proxy.returncode != 0 or any(
            proxy.returncode != 0 for proxy in remote_proxies
        ):
            raise RuntimeError(f"NPU MPI sidecar shutdown failed; see {job_dir}")
        residue = list(job_dir.rglob("*.sock"))
        if residue:
            raise RuntimeError(f"Unix socket residue remains: {residue}")
        for remote in remotes[1:]:
            residue_check = subprocess.run(
                [
                    "ssh",
                    str(remote["ssh"]),
                    shlex.join(["find", str(job_dir), "-type", "s", "-print", "-quit"]),
                ],
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
            if residue_check.returncode != 0 or residue_check.stdout.strip():
                raise RuntimeError(
                    f"rank {remote['rank']} residue verification failed: "
                    f"{residue_check.stdout}{residue_check.stderr}"
                )
        print(f"two-machine master real-NPU MPI sidecar phase-1 validation PASS; logs: {job_dir}")
        return 0
    finally:
        for process in reversed(processes):
            _terminate(process)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("one-host", "two-host", "npu"), required=True)
    parser.add_argument("--config", required=True)
    ns = parser.parse_args(argv)
    config = _load_config(ns.config)
    if ns.mode == "one-host":
        return run_one_host(config)
    if ns.mode == "two-host":
        return run_two_host(config)
    return run_npu(config)


if __name__ == "__main__":
    sys.exit(main())
