# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path


def _load_launcher():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "mpi_l4_sidecar_launch", root / "tools/mpi_l4_sidecar/launch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_managed_sidecar_command_uses_mpi_rank_local_proxy(tmp_path):
    launcher = _load_launcher()
    bootstrap = tmp_path / "bootstrap.sock"
    command = launcher._managed_sidecar_command(
        mpi_command=["mpirun", "-np", "2", "-hosts", "host-a,host-b"],
        sidecar_binary="/shared/simpler-mpi-l4-sidecar",
        python="/shared/.venv/bin/python",
        proxy_template=str(tmp_path / "proxy.%r.sock"),
        session_dir_template=str(tmp_path / "sessions.%r"),
        bootstrap_path=bootstrap,
        topology_id="host-a-host-b",
        worker_map="0:0;1:1",
        timeout_s=12.5,
    )

    assert command[:5] == ["mpirun", "-np", "2", "-hosts", "host-a,host-b"]
    assert "--manage-proxy" in command
    assert command[command.index("--proxy-python") + 1] == "/shared/.venv/bin/python"
    assert command[command.index("--bootstrap-socket") + 1] == str(bootstrap)
    assert command[command.index("--proxy-startup-timeout-ms") + 1] == "12500"
    assert "ssh" not in command


def test_mpich_npu_command_is_derived_from_rank_hosts(tmp_path):
    launcher = _load_launcher()
    config = {"mpi": {"implementation": "mpich", "launcher": "/opt/mpich/bin/mpirun"}}
    remotes = [
        {"rank": 0, "mpi_host": "120.9.10.37"},
        {"rank": 1, "mpi_host": "120.9.10.35"},
    ]

    command = launcher._derived_npu_mpi_command(config, remotes, tmp_path)

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


def test_logged_command_is_mirrored_live_and_captured(tmp_path, capsys):
    launcher = _load_launcher()
    log = tmp_path / "child.log"

    result = launcher._run_logged(
        [sys.executable, "-c", 'print("MPI bootstrap READY", flush=True)'],
        env=dict(os.environ),
        log_path=log,
        timeout_s=2.0,
        label="test child",
    )

    assert result.returncode == 0
    assert result.stdout == "MPI bootstrap READY\n"
    assert log.read_text(encoding="utf-8") == result.stdout
    assert "MPI bootstrap READY" in capsys.readouterr().out


def test_npu_case_routes_group_to_both_local_l2_workers():
    root = Path(__file__).resolve().parents[3]
    source = (root / "tools/mpi_l4_sidecar/npu_e2e_case.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "remote_l3_group_orch"
    )
    submit = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit_next_level_group"
    )
    workers = next(keyword.value for keyword in submit.keywords if keyword.arg == "workers")

    assert ast.literal_eval(workers) == [0, 1]


def test_frame_records_accept_mpi_prefixed_proxy_output(tmp_path):
    launcher = _load_launcher()
    log = tmp_path / "sidecar.log"
    log.write_text(
        '[1] {"event":"L4_TO_MPI","lane":"COMMAND","session_id":7,'
        '"worker_id":1,"frame_type":2,"sequence":3,"sha256":"abc"}\n',
        encoding="utf-8",
    )

    assert launcher._frame_records(log, "L4_TO_MPI") == {(7, 1, 2, 3, "abc"): 1}
