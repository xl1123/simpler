# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Launch two MPI L3 ranks and run the real cross-machine L4 NPU case."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _status(message: str) -> None:
    print(f"[mpi-l3] {message}", flush=True)


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("topology config must be a JSON object")
    return value


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    entries = [str(REPO_ROOT), str(REPO_ROOT / "python")]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _validate(config: dict[str, Any]) -> tuple[str, str, float, list[dict[str, Any]]]:
    python = os.path.abspath(str(config["python"]))
    library = os.path.abspath(str(config["mpi_transport_library"]))
    timeout_s = float(config.get("timeout_s", 300.0))
    if not os.path.isfile(python) or not os.access(python, os.X_OK):
        raise ValueError(f"configured Python is not executable: {python}")
    if not os.path.isfile(library):
        raise ValueError(
            f"MPI L3 transport library does not exist: {library}; run tools/mpi_l3/build.sh"
        )
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    hosts = config.get("hosts")
    if not isinstance(hosts, list) or len(hosts) != 2:
        raise ValueError("direct MPI L3 requires exactly two host entries")
    by_rank = {int(host["rank"]): dict(host) for host in hosts}
    if set(by_rank) != {0, 1} or not bool(by_rank[0].get("local", False)):
        raise ValueError("hosts must contain local rank 0 and remote rank 1")
    ordered = [by_rank[0], by_rank[1]]
    for host in ordered:
        if not str(host.get("mpi_host", "")):
            raise ValueError(f"rank {host['rank']} requires mpi_host")
        worker_id = int(host.get("worker_id", host["rank"]))
        if worker_id != int(host["rank"]):
            raise ValueError("phase-1 topology requires worker_id to equal MPI rank")
        host["worker_id"] = worker_id
        devices = host.get("device_ids")
        if not isinstance(devices, list) or len(devices) != 2:
            raise ValueError(f"rank {host['rank']} requires exactly two device_ids")
    return python, library, timeout_s, ordered


def _mpi_command(config: dict[str, Any], hosts: list[dict[str, Any]], job_dir: Path) -> list[str]:
    explicit = config.get("mpi_command")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("mpi_command must be a non-empty string list")
        return [str(item) for item in explicit]
    mpi = config.get("mpi")
    if not isinstance(mpi, dict):
        raise ValueError("mpi must be an object when mpi_command is not set")
    launcher = str(mpi.get("launcher", "mpirun"))
    implementation = str(mpi.get("implementation", "mpich")).lower()
    host_names = [str(host["mpi_host"]) for host in hosts]
    if implementation == "mpich":
        hostfile = job_dir / "mpi.hostfile"
        hostfile.write_text("".join(f"{host}\n" for host in host_names), encoding="utf-8")
        return [launcher, "-f", str(hostfile), "-ppn", "1", "-np", "2"]
    if implementation == "openmpi":
        return [launcher, "--host", ",".join(f"{host}:1" for host in host_names), "-np", "2"]
    raise ValueError("mpi.implementation must be 'mpich' or 'openmpi'")


def _follow_log(path: Path, process: subprocess.Popen[Any]) -> threading.Thread:
    def follow() -> None:
        position = 0
        while True:
            if path.exists():
                with path.open(encoding="utf-8", errors="replace") as stream:
                    stream.seek(position)
                    chunk = stream.read()
                    position = stream.tell()
                if chunk:
                    print(chunk, end="", flush=True)
            if process.poll() is not None:
                return
            time.sleep(0.05)

    thread = threading.Thread(target=follow, name="mpi-l3-log-follower", daemon=True)
    thread.start()
    return thread


def _wait_gateway(path: Path, process: subprocess.Popen[Any], log: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="replace")[-8192:] if log.exists() else ""
            raise RuntimeError(f"MPI L3 world exited with {process.returncode} before gateway READY:\n{tail}")
        time.sleep(0.05)
    raise TimeoutError(f"rank 0 gateway {path} was not created within {timeout_s:g}s")


def _run_checked(command: list[str], env: dict[str, str], timeout_s: float, label: str) -> str:
    _status(f"{label}: {shlex.join(command)}")
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with {result.returncode}")
    return result.stdout


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _verify_log(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    required = {
        '"event": "MPI_INIT_AFTER_L2_FORK"': 2,
        '"event": "MPI_L3_WORLD_READY"': 2,
        '"event": "L4_MPI_L3_GATEWAY_READY"': 1,
        '"event": "MPI_L3_SESSION_READY"': 2,
        '"event": "MPI_L3_SEND"': 1,
        '"event": "MPI_L3_RECV"': 1,
    }
    missing = [f"{event} x{count}" for event, count in required.items() if text.count(event) < count]
    if missing:
        raise RuntimeError(f"MPI L3 log evidence is incomplete: {', '.join(missing)}; see {path}")
    records: dict[str, Counter[tuple[Any, ...]]] = {
        event: Counter() for event in ("L4_TO_MPI", "MPI_TO_L3", "L3_TO_MPI", "MPI_TO_L4")
    }
    for line in text.splitlines():
        json_start = line.find("{")
        if json_start < 0:
            continue
        try:
            record = json.loads(line[json_start:])
        except json.JSONDecodeError:
            continue
        event = record.get("event")
        if event not in records or record.get("lane") != "COMMAND":
            continue
        key = (
            int(record["session_id"]),
            int(record["worker_id"]),
            int(record["frame_type"]),
            int(record["sequence"]),
            str(record["sha256"]),
        )
        records[event][key] += 1
    if records["L4_TO_MPI"] != records["MPI_TO_L3"]:
        raise RuntimeError("L4->L3 direct MPI command frame sequence/hash comparison failed")
    if records["L3_TO_MPI"] != records["MPI_TO_L4"]:
        raise RuntimeError("L3->L4 direct MPI command frame sequence/hash comparison failed")
    if not any(key[2] == 2 for key in records["L4_TO_MPI"]):
        raise RuntimeError("direct MPI L3 log is missing a TASK command frame")
    if not any(key[2] == 5 for key in records["L3_TO_MPI"]):
        raise RuntimeError("direct MPI L3 log is missing a COMPLETION command frame")
    _status(
        "command frame sequence/hash comparison PASS "
        f"(L4->L3={sum(records['L4_TO_MPI'].values())}, "
        f"L3->L4={sum(records['L3_TO_MPI'].values())})"
    )


def run(config_path: str) -> int:
    config_path = os.path.abspath(config_path)
    config = _load_config(config_path)
    python, library, timeout_s, hosts = _validate(config)
    work_dir = Path(str(config.get("work_dir", "/tmp")))
    job_dir = work_dir / f"simpler-mpi-l3-{uuid.uuid4().hex[:12]}"
    job_dir.mkdir(mode=0o700, parents=True)
    gateway = job_dir / "l4-gateway.sock"
    mpi_log = job_dir / "mpi-l3.log"
    env = _python_env()
    mpi = _mpi_command(config, hosts, job_dir)
    rank_command = [
        python,
        "-m",
        "simpler.mpi_l3_worker",
        "--config",
        config_path,
        "--mpi-library",
        library,
        "--bootstrap-socket",
        str(gateway),
        "--session-dir",
        str(job_dir / "sessions"),
    ]
    command = mpi + rank_command
    _status("execution mode: direct MPI L3; remote_l3_worker TCP daemons are not used")
    _status(f"job directory: {job_dir}")
    _status(f"MPI command: {shlex.join(command)}")
    process: subprocess.Popen[Any] | None = None
    log_stream = mpi_log.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        follower = _follow_log(mpi_log, process)
        _wait_gateway(gateway, process, mpi_log, timeout_s)
        _status("rank 0 UDS gateway READY; starting L4 NPU case")
        case_command = [
            python,
            str(REPO_ROOT / "tools/mpi_l3/npu_e2e_case.py"),
            "--control-transport",
            "mpi_l3",
            "--machine-a",
            "mpi://rank/0",
            "--machine-b",
            "mpi://rank/1",
            "--machine-a-mpi-rank",
            "0",
            "--machine-b-mpi-rank",
            "1",
            "--machine-a-devices",
            ",".join(str(item) for item in hosts[0]["device_ids"]),
            "--machine-b-devices",
            ",".join(str(item) for item in hosts[1]["device_ids"]),
            "--gateway-endpoint",
            str(gateway),
            "--platform",
            str(config.get("platform", "a2a3")),
            "--runtime",
            str(config.get("runtime", "tensormap_and_ringbuffer")),
            "--block-dim",
            str(int(config.get("block_dim", 1))),
            "--timeout",
            str(timeout_s),
        ]
        output = _run_checked(case_command, env, timeout_s, "L4 NPU validation")
        if '"status": "PASS"' not in output or '"control_transport": "mpi_l3"' not in output:
            raise RuntimeError("L4 NPU validation did not emit the direct MPI L3 PASS record")
        _run_checked(
            [python, "-m", "simpler.mpi_l3_worker", "--stop-world", str(gateway)],
            env,
            min(timeout_s, 30.0),
            "MPI L3 shutdown",
        )
        process.wait(timeout=timeout_s)
        follower.join(timeout=2.0)
        log_stream.flush()
        if process.returncode != 0:
            raise RuntimeError(f"MPI L3 world exited with {process.returncode}; see {mpi_log}")
        _verify_log(mpi_log)
        residue = list(job_dir.rglob("*.sock"))
        if residue:
            raise RuntimeError(f"Unix socket residue remains: {residue}")
        _status(f"direct MPI L3 validation PASS; logs: {job_dir}")
        return 0
    finally:
        if process is not None:
            _terminate(process)
        log_stream.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    ns = parser.parse_args(argv)
    return run(ns.config)


if __name__ == "__main__":
    sys.exit(main())
